# hca_protonet.py - Complete Corrected Version
import torch
import torch.nn as nn
import torch.nn.functional as F

class HCAProtoNet(nn.Module):
    def __init__(self, backbone, num_classes=5, 
                 K_shared=20, rare_classes=None, K_rare_c=5,
                 m_shared=0.7, m_rare=0.9,
                 lambda_div=0.01, lambda_sep=0.05, lambda_cov=0.1, lambda_ent=0.01,
                 threshold_freq=0.1, min_active=2, temperature=1.5, eps=1e-8):
        super(HCAProtoNet, self).__init__()
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

        # Shared prototypes
        self.shared_prototypes = None
        self.W_shared_to_class = None

        # Rare prototypes - use ParameterDict for proper registration
        self.rare_prototypes = nn.ParameterDict()

        # Training statistics (set externally)
        self.training_frequencies = None
        self.N_max = None

    def _init_prototypes(self, feature_dim):
        device = next(self.backbone.parameters()).device
        
        self.shared_prototypes = nn.Parameter(
            torch.rand(self.K_shared, feature_dim, device=device)
        )
        self.W_shared_to_class = nn.Parameter(
            torch.randn(self.K_shared, self.num_classes, device=device) * 0.1
        )

        # Properly register rare prototypes
        for c in self.rare_classes:
            self.rare_prototypes[str(c)] = nn.Parameter(
                torch.rand(self.K_rare_c, feature_dim, device=device)
            )

    def forward(self, x, target=None):
        features = self.backbone(x)  # B x D
        B, D = features.shape

        if self.shared_prototypes is None:
            self._init_prototypes(D)

        # Shared prototypes
        shared_sim = F.cosine_similarity(
            features.unsqueeze(1), 
            self.shared_prototypes.unsqueeze(0), 
            dim=2
        )  # B x K_shared
        logits_shared = shared_sim @ self.W_shared_to_class  # B x C

        # Temperature-scaled uncertainty
        probs_shared = F.softmax(logits_shared / self.temperature, dim=-1)
        uncertainty = -torch.sum(
            probs_shared * torch.log(probs_shared + self.eps), 
            dim=-1, 
            keepdim=True
        )
        uncertainty = uncertainty / torch.log(
            torch.tensor(self.num_classes, dtype=torch.float32, device=features.device)
        )

        # Rarity factor
        rarity_factor = torch.zeros(self.num_classes, device=features.device)
        if self.training_frequencies is not None and self.N_max is not None:
            for c in self.rare_classes:
                rarity_factor[c] = torch.log(
                    torch.tensor(
                        self.N_max / (self.training_frequencies[c] + self.eps), 
                        device=features.device
                    )
                )

        # Rare-class prototypes
        logits_rare = torch.zeros(B, self.num_classes, device=features.device)
        
        if target is not None:
            # Training mode
            for c in self.rare_classes:
                rare_proto = self.rare_prototypes[str(c)]
                rare_sim_c = F.cosine_similarity(
                    features.unsqueeze(1), 
                    rare_proto.unsqueeze(0), 
                    dim=2
                )  # B x K_rare_c
                
                # Ensure proper shape
                is_class_c = (target == c).float().view(-1, 1)
                gate_c = rarity_factor[c] * uncertainty * is_class_c
                
                # Direct aggregation
                logits_rare[:, c] = gate_c.squeeze() * rare_sim_c.sum(dim=-1)
        else:
            # Inference mode
            for c in self.rare_classes:
                rare_proto = self.rare_prototypes[str(c)]
                rare_sim_c = F.cosine_similarity(
                    features.unsqueeze(1), 
                    rare_proto.unsqueeze(0), 
                    dim=2
                )
                gate_c = rarity_factor[c] * uncertainty
                logits_rare[:, c] = gate_c.squeeze() * rare_sim_c.sum(dim=-1)

        logits = logits_shared + logits_rare
        return logits, features, shared_sim, logits_rare

    def compute_loss(self, logits, features, target):
        # Cross-entropy with class weights
        class_weights = None
        if self.training_frequencies is not None and self.N_max is not None:
            weights = [
                self.N_max / (self.training_frequencies.get(c, 1) + self.eps) 
                for c in range(self.num_classes)
            ]
            class_weights = torch.tensor(weights, device=logits.device, dtype=torch.float32)
        
        ce_loss = F.cross_entropy(logits, target, weight=class_weights)

        # Diversity loss
        all_protos = [self.shared_prototypes]
        for c in self.rare_classes:
            all_protos.append(self.rare_prototypes[str(c)])
        
        all_protos_flat = torch.cat(all_protos, dim=0)  # total_K x D
        sim_matrix = F.cosine_similarity(
            all_protos_flat.unsqueeze(0), 
            all_protos_flat.unsqueeze(1), 
            dim=2
        )
        # Mask diagonal
        mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        diversity_loss = self.lambda_div * sim_matrix[mask].mean()

        # Separation loss
        separation_loss = 0
        if len(self.rare_classes) > 0:
            for c in self.rare_classes:
                sep = F.cosine_similarity(
                    self.rare_prototypes[str(c)].unsqueeze(1), 
                    self.shared_prototypes.unsqueeze(0), 
                    dim=2
                ).mean()
                separation_loss += sep
            separation_loss = self.lambda_sep * separation_loss / len(self.rare_classes)

        # Coverage loss
        coverage_loss = 0
        if len(self.rare_classes) > 0:
            for c in self.rare_classes:
                mask = (target == c)
                if mask.sum() > 0:
                    rare_proto = self.rare_prototypes[str(c)]
                    rare_sim_c = F.cosine_similarity(
                        features[mask].unsqueeze(1), 
                        rare_proto.unsqueeze(0), 
                        dim=2
                    )
                    active_count = (rare_sim_c.max(dim=0)[0] > 0.5).float().sum()
                    coverage_loss += F.relu(self.min_active - active_count)
            coverage_loss = self.lambda_cov * coverage_loss / len(self.rare_classes)

        # Entropy regularization
        probs = F.softmax(logits / self.temperature, dim=-1)
        entropy_reg = -self.lambda_ent * torch.sum(
            probs * torch.log(probs + self.eps), 
            dim=-1
        ).mean()

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
        # Shared prototypes with cosine similarity
        shared_sim = F.cosine_similarity(
            features.unsqueeze(1), 
            self.shared_prototypes.unsqueeze(0), 
            dim=2
        )
        assign_shared_idx = shared_sim.argmax(dim=-1)  # B
        
        for i in range(self.K_shared):
            mask = (assign_shared_idx == i)
            if mask.sum() > 0:
                centroid = features[mask].mean(dim=0)
                self.shared_prototypes[i].data = (
                    self.m_shared * self.shared_prototypes[i].data + 
                    (1 - self.m_shared) * centroid
                )

        # Rare prototypes: class-conditional
        for c in self.rare_classes:
            mask_class = (target == c)
            if mask_class.sum() > 0:
                feats_c = features[mask_class]
                rare_proto = self.rare_prototypes[str(c)]
                
                rare_sim = F.cosine_similarity(
                    feats_c.unsqueeze(1), 
                    rare_proto.unsqueeze(0), 
                    dim=2
                )
                assign_idx = rare_sim.argmax(dim=-1)
                
                for i in range(self.K_rare_c):
                    mask_proto = (assign_idx == i)
                    if mask_proto.sum() > 0:
                        centroid = feats_c[mask_proto].mean(dim=0)
                        self.rare_prototypes[str(c)][i].data = (
                            self.m_rare * self.rare_prototypes[str(c)][i].data + 
                            (1 - self.m_rare) * centroid
                        )