import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import Counter
import warnings


class HCAProtoNet(nn.Module):
    def __init__(self, backbone, num_classes=5, 
                 K_shared=20, rare_classes=None, K_rare_c=5,
                 m_shared=0.7, m_rare=0.9,
                 lambda_div=0.01, lambda_sep=0.05, lambda_cov=0.1, lambda_ent=0.01,
                 threshold_freq=0.1, min_active=2, temperature=1.5, eps=1e-8):
        super(HCAProtoNet, self).__init__()
        
        # Validation
        assert num_classes > 0, "num_classes must be positive"
        assert K_shared > 0, "K_shared must be positive"
        assert K_rare_c > 0, "K_rare_c must be positive"
        assert 0 < m_shared < 1, "m_shared must be in (0, 1)"
        assert 0 < m_rare < 1, "m_rare must be in (0, 1)"
        
        self.backbone = backbone
        self.num_classes = num_classes
        self.K_shared = K_shared
        self.rare_classes = rare_classes if rare_classes is not None else []
        self.K_rare_c = K_rare_c
        self.m_shared = m_shared
        self.m_rare = m_rare
        self.lambda_div = lambda_div
        self.lambda_sep = lambda_sep
        self.lambda_cov = lambda_cov
        self.lambda_ent = lambda_ent
        self.threshold_freq = threshold_freq
        self.min_active = min_active
        self.temperature = temperature
        self.eps = eps
        
        # Validate rare classes
        for c in self.rare_classes:
            assert 0 <= c < num_classes, f"Rare class {c} out of range [0, {num_classes})"

        # Shared prototypes
        self.shared_prototypes = None
        self.W_shared_to_class = None

        # Rare prototypes
        self.rare_prototypes = nn.ParameterDict()

        # Training statistics (must be set via initialize_from_dataset)
        self.training_frequencies = None
        self.N_max = None
        self.rarity_factor = None  # Precomputed for efficiency
        self._log_num_classes = None  # Cached for efficiency
        self._is_initialized = False

    def initialize_from_dataset(self, dataset):
        """
        Initialize training statistics and prototypes from dataset.
        MUST be called before training.
        """
        print("Initializing model from dataset...")
        
        # Compute class frequencies
        label_counts = Counter()
        all_features = []
        all_labels = []
        
        device = next(self.backbone.parameters()).device
        
        with torch.no_grad():
            # Sample features for prototype initialization
            sample_size = min(len(dataset), 5000)  # Don't use entire dataset for efficiency
            indices = torch.randperm(len(dataset))[:sample_size]
            
            for idx in indices:
                data = dataset[idx]
                label = data['primus_label']
                if torch.is_tensor(label):
                    label = label.item()
                label_counts[label] += 1
                
                # Extract features for k-means initialization
                if len(all_features) < 1000:  # Limit for memory
                    x = data['bmode'].unsqueeze(0).to(device)
                    feat = self.backbone(x).cpu()
                    all_features.append(feat)
                    all_labels.append(label)
        
        # Set frequencies
        self.training_frequencies = {
            c: label_counts.get(c, 1) for c in range(self.num_classes)
        }
        self.N_max = max(self.training_frequencies.values())
        
        # Validate rare classes
        for c in self.rare_classes:
            freq = self.training_frequencies[c] / self.N_max
            if freq > 0.3:
                warnings.warn(
                    f"Class {c} marked as rare but has frequency {freq:.2%} "
                    "(typically rare classes should have <30% of max class frequency)"
                )
        
        print(f"Training frequencies: {self.training_frequencies}")
        print(f"N_max: {self.N_max}")
        
        # Precompute rarity factors (avoid recomputing every forward pass)
        self.rarity_factor = torch.zeros(self.num_classes, device=device)
        for c in self.rare_classes:
            self.rarity_factor[c] = torch.log(
                torch.tensor(
                    self.N_max / (self.training_frequencies[c] + self.eps),
                    device=device
                )
            )
        print(f"Rarity factors: {self.rarity_factor.tolist()}")
        
        # Initialize prototypes using k-means on sampled features
        if len(all_features) > 0:
            all_features = torch.cat(all_features, dim=0)  # [N, D]
            all_labels = torch.tensor(all_labels)
            self._init_prototypes_from_features(all_features, all_labels)
        else:
            # Fallback to random initialization
            feat_dim = self.backbone(torch.randn(1, *dataset[0]['bmode'].shape).to(device)).shape[1]
            self._init_prototypes(feat_dim)

    def _init_prototypes_from_features(self, features, labels):
        """Initialize prototypes using k-means clustering on real features."""
        device = next(self.backbone.parameters()).device
        features = features.to(device)
        labels = labels.to(device)
        D = features.shape[1]
        
        print(f"Initializing prototypes from {len(features)} features...")
        
        # Shared prototypes: k-means on all features
        shared_protos = self._kmeans_init(features, self.K_shared)
        self.shared_prototypes = nn.Parameter(shared_protos)
        
        # Classification weights
        self.W_shared_to_class = nn.Parameter(
            torch.randn(self.K_shared, self.num_classes, device=device) * 0.01
        )
        
        # Rare prototypes: k-means per class
        for c in self.rare_classes:
            mask = (labels == c)
            if mask.sum() >= self.K_rare_c:
                feats_c = features[mask]
                rare_protos = self._kmeans_init(feats_c, self.K_rare_c)
            else:
                # Not enough samples, use random subset
                rare_protos = features[mask][torch.randperm(mask.sum())[:self.K_rare_c]]
                if len(rare_protos) < self.K_rare_c:
                    # Pad with noise
                    padding = torch.randn(self.K_rare_c - len(rare_protos), D, device=device) * 0.01
                    rare_protos = torch.cat([rare_protos, padding], dim=0)
            
            self.rare_prototypes[str(c)] = nn.Parameter(rare_protos)
        
        # Cache log(num_classes)
        self._log_num_classes = torch.log(
            torch.tensor(self.num_classes, dtype=torch.float32, device=device)
        )
        
        self._is_initialized = True
        print(f"Prototypes initialized: shared={self.K_shared}, rare={len(self.rare_classes)}x{self.K_rare_c}")

    def _kmeans_init(self, features, K, max_iters=10):
        """Simple k-means for prototype initialization."""
        N, D = features.shape
        device = features.device
        
        # Initialize centroids randomly from data
        indices = torch.randperm(N, device=device)[:K]
        centroids = features[indices].clone()
        
        for _ in range(max_iters):
            # Assign to nearest centroid
            dists = torch.cdist(features, centroids)
            assignments = dists.argmin(dim=1)
            
            # Update centroids
            for k in range(K):
                mask = (assignments == k)
                if mask.sum() > 0:
                    centroids[k] = features[mask].mean(dim=0)
        
        return centroids

    def _init_prototypes(self, feature_dim):
        """Fallback: Xavier initialization if no features available."""
        device = next(self.backbone.parameters()).device
        
        self.shared_prototypes = nn.Parameter(
            torch.empty(self.K_shared, feature_dim, device=device)
        )
        nn.init.xavier_normal_(self.shared_prototypes)
        
        self.W_shared_to_class = nn.Parameter(
            torch.randn(self.K_shared, self.num_classes, device=device) * 0.01
        )

        for c in self.rare_classes:
            proto = nn.Parameter(
                torch.empty(self.K_rare_c, feature_dim, device=device)
            )
            nn.init.xavier_normal_(proto)
            self.rare_prototypes[str(c)] = proto
        
        self._log_num_classes = torch.log(
            torch.tensor(self.num_classes, dtype=torch.float32, device=device)
        )
        
        self._is_initialized = True

    def forward(self, x, target=None):
        if not self._is_initialized:
            raise RuntimeError(
                "Model not initialized! Call initialize_from_dataset() before forward pass."
            )
        
        features = self.backbone(x)  # B x D
        B, D = features.shape

        # EFFICIENCY: Normalize once for all similarity computations
        features_norm = F.normalize(features, p=2, dim=1)
        shared_proto_norm = F.normalize(self.shared_prototypes, p=2, dim=1)
        
        # Shared prototypes (optimized with matrix multiply)
        shared_sim = features_norm @ shared_proto_norm.T  # B x K_shared
        logits_shared = shared_sim @ self.W_shared_to_class  # B x C

        # Temperature-scaled uncertainty (use cached log_num_classes)
        probs_shared = F.softmax(logits_shared / self.temperature, dim=-1)
        entropy = -torch.sum(probs_shared * torch.log(probs_shared + self.eps), dim=-1)
        uncertainty = entropy / self._log_num_classes  # B

        # Rare-class prototypes
        logits_rare = torch.zeros(B, self.num_classes, device=features.device)
        
        # FIX: Consistent gating for train and inference (no train/test mismatch)
        for c in self.rare_classes:
            rare_proto_norm = F.normalize(self.rare_prototypes[str(c)], p=2, dim=1)
            rare_sim_c = features_norm @ rare_proto_norm.T  # B x K_rare_c
            
            # Use max similarity (more stable than sum)
            rare_sim_max = rare_sim_c.max(dim=1)[0]  # B
            
            # Gate based on uncertainty (same for train/inference)
            gate_c = self.rarity_factor[c] * uncertainty  # B
            logits_rare[:, c] = gate_c * rare_sim_max

        logits = logits_shared + logits_rare
        return logits, features, shared_sim, logits_rare

    def compute_loss(self, logits, features, target):
        device = logits.device
        
        # Cross-entropy with normalized class weights
        if self.training_frequencies is not None and self.N_max is not None:
            weights = torch.tensor([
                self.N_max / (self.training_frequencies.get(c, 1) + self.eps) 
                for c in range(self.num_classes)
            ], device=device, dtype=torch.float32)
            # Normalize to prevent loss explosion
            weights = weights / weights.sum() * self.num_classes
        else:
            weights = None
        
        ce_loss = F.cross_entropy(logits, target, weight=weights)

        # Diversity loss (OPTIMIZED: normalize once, use matrix multiply)
        diversity_loss = torch.tensor(0.0, device=device)
        if self.lambda_div > 0:
            all_protos_norm = [F.normalize(self.shared_prototypes, p=2, dim=1)]
            for c in self.rare_classes:
                all_protos_norm.append(F.normalize(self.rare_prototypes[str(c)], p=2, dim=1))
            
            all_protos_flat = torch.cat(all_protos_norm, dim=0)  # total_K x D
            sim_matrix = all_protos_flat @ all_protos_flat.T  # total_K x total_K
            
            # Mask diagonal
            K_total = sim_matrix.size(0)
            mask = ~torch.eye(K_total, dtype=torch.bool, device=device)
            diversity_loss = self.lambda_div * sim_matrix[mask].mean()

        # Separation loss (OPTIMIZED)
        separation_loss = torch.tensor(0.0, device=device)
        if self.lambda_sep > 0 and len(self.rare_classes) > 0:
            shared_norm = F.normalize(self.shared_prototypes, p=2, dim=1)
            for c in self.rare_classes:
                rare_norm = F.normalize(self.rare_prototypes[str(c)], p=2, dim=1)
                sep = (rare_norm @ shared_norm.T).mean()
                separation_loss += sep
            separation_loss = self.lambda_sep * separation_loss / len(self.rare_classes)

        # Coverage loss
        coverage_loss = torch.tensor(0.0, device=device)
        if self.lambda_cov > 0 and len(self.rare_classes) > 0:
            features_norm = F.normalize(features, p=2, dim=1)
            for c in self.rare_classes:
                mask = (target == c)
                if mask.sum() > 0:
                    rare_norm = F.normalize(self.rare_prototypes[str(c)], p=2, dim=1)
                    rare_sim_c = features_norm[mask] @ rare_norm.T
                    
                    # Soft threshold (more stable)
                    active_count = (rare_sim_c.max(dim=0)[0] > 0.3).float().sum()
                    coverage_loss += F.relu(self.min_active - active_count)
            coverage_loss = self.lambda_cov * coverage_loss / len(self.rare_classes)

        # Entropy regularization
        entropy_reg = torch.tensor(0.0, device=device)
        if self.lambda_ent > 0:
            probs = F.softmax(logits / self.temperature, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + self.eps), dim=-1)
            entropy_reg = -self.lambda_ent * entropy.mean()

        total_loss = ce_loss + diversity_loss + separation_loss + coverage_loss + entropy_reg
        
        return total_loss, {
            'ce': ce_loss.item(),
            'diversity': diversity_loss.item(),
            'separation': separation_loss.item(),
            'coverage': coverage_loss.item(),
            'entropy': entropy_reg.item()
        }

    @torch.no_grad()
    def update_prototypes(self, features, target):
        """Momentum-based prototype updates (OPTIMIZED with normalization)."""
        # Normalize for stable updates
        features_norm = F.normalize(features, p=2, dim=1)
        shared_proto_norm = F.normalize(self.shared_prototypes, p=2, dim=1)
        
        # Shared prototypes
        shared_sim = features_norm @ shared_proto_norm.T  # B x K_shared
        assign_shared_idx = shared_sim.argmax(dim=-1)  # B
        
        for i in range(self.K_shared):
            mask = (assign_shared_idx == i)
            if mask.sum() > 0:
                centroid = features[mask].mean(dim=0)
                centroid_norm = F.normalize(centroid.unsqueeze(0), p=2, dim=1).squeeze(0)
                self.shared_prototypes[i].data = (
                    self.m_shared * self.shared_prototypes[i].data + 
                    (1 - self.m_shared) * centroid_norm
                )

        # Rare prototypes: class-conditional
        for c in self.rare_classes:
            mask_class = (target == c)
            if mask_class.sum() > 0:
                feats_c = features[mask_class]
                feats_c_norm = F.normalize(feats_c, p=2, dim=1)
                rare_proto_norm = F.normalize(self.rare_prototypes[str(c)], p=2, dim=1)
                
                rare_sim = feats_c_norm @ rare_proto_norm.T
                assign_idx = rare_sim.argmax(dim=-1)
                
                for i in range(self.K_rare_c):
                    mask_proto = (assign_idx == i)
                    if mask_proto.sum() > 0:
                        centroid = feats_c[mask_proto].mean(dim=0)
                        centroid_norm = F.normalize(centroid.unsqueeze(0), p=2, dim=1).squeeze(0)
                        self.rare_prototypes[str(c)][i].data = (
                            self.m_rare * self.rare_prototypes[str(c)][i].data + 
                            (1 - self.m_rare) * centroid_norm
                        )