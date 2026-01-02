import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model
from timm.models.vision_transformer import VisionTransformer, _cfg

class ViTBackbone(nn.Module):
    """
    Enhanced ViT Backbone optimized for prototype learning
    
    Key improvements over ProtoPNet/ProtoViT:
    1. Flexible feature extraction (CLS, patch tokens, or hybrid)
    2. Multi-scale feature aggregation
    3. Learnable projection head for better prototype alignment
    4. Feature normalization for stable similarity computation
    5. Spatial information preservation
    """
    def __init__(self, 
                 features, 
                 img_size,
                 output_dim=512,
                 pooling_mode='gap',  # 'gap', 'cls', 'hybrid', 'attention'
                 use_projection=True,
                 normalize_features=True,
                 multi_scale=False):
        super().__init__()
        self.features = features
        self.img_size = img_size
        self.pooling_mode = pooling_mode
        self.normalize_features = normalize_features
        self.multi_scale = multi_scale
        
        # Detect architecture
        features_name = str(features).upper()
        if 'VISION' in features_name or 'DEIT' in features_name:
            self.arc = 'deit'
        elif 'DINOV2' in features_name:
            self.arc = 'dinov2'
        elif 'DINOV3' in features_name or 'DINO' in features_name:
            self.arc = 'dinov3'
        else:
            # Fallback: try to detect by checking attributes
            if hasattr(features, 'patch_embed'):
                self.arc = 'deit'  # Default assumption
            else:
                raise ValueError(f"Unknown ViT architecture: {features_name}")
        
        # Get embedding dimension from the model
        if hasattr(features, 'embed_dim'):
            self.embed_dim = features.embed_dim
        elif hasattr(features, 'num_features'):
            self.embed_dim = features.num_features
        else:
            # Try to infer from cls_token
            self.embed_dim = features.cls_token.shape[-1]
        
        # Projection head for prototype learning
        if use_projection:
            if multi_scale:
                # Multi-scale: combine features from different layers
                self.projection = nn.Sequential(
                    nn.Linear(self.embed_dim * 2, self.embed_dim),  # *2 for multi-scale
                    nn.LayerNorm(self.embed_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.embed_dim, output_dim),
                    nn.LayerNorm(output_dim)
                )
            else:
                self.projection = nn.Sequential(
                    nn.Linear(self.embed_dim, self.embed_dim),
                    nn.LayerNorm(self.embed_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.embed_dim, output_dim),
                    nn.LayerNorm(output_dim)
                )
        else:
            # Identity projection
            if self.embed_dim != output_dim:
                self.projection = nn.Linear(self.embed_dim, output_dim)
            else:
                self.projection = nn.Identity()
        
        # Attention pooling (learnable)
        if pooling_mode == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim // 4),
                nn.Tanh(),
                nn.Linear(self.embed_dim // 4, 1)
            )

    def forward(self, x):
        """
        Forward pass with flexible feature extraction
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            features: Extracted features [B, output_dim]
        """
        # Extract patch embeddings
        x = self.features.patch_embed(x)
        cls_token = self.features.cls_token.expand(x.shape[0], -1, -1)

        # Forward through transformer blocks
        if self.arc == 'deit':
            x = torch.cat((cls_token, x), dim=1)
            x = self.features.pos_drop(x + self.features.pos_embed)
            
            # Store intermediate features for multi-scale
            if self.multi_scale:
                mid_idx = len(self.features.blocks) // 2
                for i, blk in enumerate(self.features.blocks):
                    x = blk(x)
                    if i == mid_idx:
                        x_mid = self.features.norm(x)  # Mid-layer features
            else:
                x = self.features.blocks(x)
            
            x = self.features.norm(x)

        elif self.arc in ['dinov2', 'dinov3']:
            # DINO models
            x = x.reshape(x.size(0), -1, x.size(-1))
            x = torch.cat((cls_token, x), dim=1)
            
            if self.multi_scale:
                mid_idx = len(self.features.blocks) // 2
                for i, blk in enumerate(self.features.blocks):
                    x = blk(x)
                    if i == mid_idx:
                        x_mid = self.features.norm(x)
            else:
                for blk in self.features.blocks:
                    x = blk(x)
            
            x = self.features.norm(x)
        
        # Feature extraction based on pooling mode
        if self.pooling_mode == 'cls':
            # Use CLS token only (traditional ViT)
            features = x[:, 0]  # B x D
            
        elif self.pooling_mode == 'gap':
            # Global Average Pooling over patch tokens (better for prototypes)
            # This preserves more spatial information
            features = x[:, 1:].mean(dim=1)  # B x D
            
        elif self.pooling_mode == 'hybrid':
            # Combine CLS and GAP (best of both worlds)
            cls_feat = x[:, 0]  # B x D
            gap_feat = x[:, 1:].mean(dim=1)  # B x D
            features = (cls_feat + gap_feat) / 2  # B x D
            
        elif self.pooling_mode == 'attention':
            # Learnable attention pooling over patches
            patch_tokens = x[:, 1:]  # B x N x D
            attn_weights = self.attention_pool(patch_tokens)  # B x N x 1
            attn_weights = F.softmax(attn_weights, dim=1)
            features = (patch_tokens * attn_weights).sum(dim=1)  # B x D
            
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")
        
        # Multi-scale feature fusion
        if self.multi_scale:
            if self.pooling_mode == 'cls':
                features_mid = x_mid[:, 0]
            else:
                features_mid = x_mid[:, 1:].mean(dim=1)
            features = torch.cat([features, features_mid], dim=1)  # B x 2D
        
        # Project to output dimension
        features = self.projection(features)  # B x output_dim
        
        # L2 normalization for stable cosine similarity
        if self.normalize_features:
            features = F.normalize(features, p=2, dim=1)
        
        return features
    
    def get_patch_features(self, x):
        """
        Extract spatial patch features (useful for prototype visualization)
        
        Returns:
            patch_features: [B, N, D] where N is number of patches
            spatial_shape: (H, W) spatial dimensions
        """
        # Extract patch embeddings
        x = self.features.patch_embed(x)
        cls_token = self.features.cls_token.expand(x.shape[0], -1, -1)

        # Forward through transformer
        if self.arc == 'deit':
            x = torch.cat((cls_token, x), dim=1)
            x = self.features.pos_drop(x + self.features.pos_embed)
            x = self.features.blocks(x)
            x = self.features.norm(x)
        elif self.arc in ['dinov2', 'dinov3']:
            x = x.reshape(x.size(0), -1, x.size(-1))
            x = torch.cat((cls_token, x), dim=1)
            for blk in self.features.blocks:
                x = blk(x)
            x = self.features.norm(x)
        
        patch_tokens = x[:, 1:]  # B x N x D
        B, N, D = patch_tokens.shape
        H = W = int(N ** 0.5)
        
        return patch_tokens, (H, W)
    

def get_pretrained_weights_path(model_name):

    if model_name in ["deit_small_patch16_224", "deit_base_patch16_224", "deit_tiny_patch16_224",
            "deit_tiny_distilled_patch16_224"]:
        if model_name == "deit_small_patch16_224":
            finetune = 'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'
        elif model_name == "deit_base_patch16_224":
            finetune = 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'
        elif model_name == "deit_tiny_patch16_224":
            finetune = 'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'

    return finetune

def get_pretrained_weights(model_name, model):
    finetune = get_pretrained_weights_path(model_name)
    print(finetune)
    if finetune.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            finetune, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(finetune, map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            del checkpoint_model[k]
    # interpolate position embedding
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint_model['pos_embed'] = new_pos_embed

    model.load_state_dict(checkpoint_model, strict=False)

    return model

def deit_tiny_patch_features(pretrained=True, **kwargs):
    base_arch = 'deit_tiny_patch16_224'
    model = create_model(
    base_arch,
    pretrained=True,
    num_classes=200,
    drop_rate=0.0,
    drop_path_rate=0.1,
    drop_block_rate=None,
    img_size=224
    )
    model.default_cfg = _cfg()
    if pretrained:
        model = get_pretrained_weights(base_arch, model)
        del model.head
    return model


def deit_small_patch_features(pretrained=True, **kwargs):
    base_arch = 'deit_small_patch16_224'
    model = create_model(
    base_arch,
    pretrained=True,
    num_classes=200,
    drop_rate=0.0,
    drop_path_rate=0.1,
    drop_block_rate=None,
    img_size=224
    )
    model.default_cfg = _cfg()
    if pretrained:
        model = get_pretrained_weights(base_arch, model)
        del model.head
    return model



def dinov3_patch_features(**kwargs):
    import os
    DINOV3_CHECKPOINTS_PATH="/datasets/exactvu_pca/checkpoint_store"
    DINOV3_LIBRARY_PATH="/home/obed/projects/aip-medilab/obed/medproj/dinov3"
    REPO_DIR = DINOV3_LIBRARY_PATH
    weight_basename = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    weights = os.path.join(DINOV3_CHECKPOINTS_PATH, weight_basename)
    if not os.path.exists(weights):
        raise FileNotFoundError(f"DINOv3 checkpoint not found: {weights}")
    kwargs["weights"] = weights
    model = torch.hub.load(REPO_DIR, "dinov3_vitl16", source="local", **kwargs)
    del model.head
    return model

def create_backbone(*, cfg, **kwargs):
    backbone_type=kwargs.get("backbone_type", getattr(cfg, "backbone_type", "deit_small_patch16_224"))
    img_size=kwargs.get("img_size", getattr(cfg, "img_size", 224))
    output_dim=kwargs.get("output_dim", getattr(cfg, "output_dim", 384))
    pooling_mode=kwargs.get("pooling_mode", getattr(cfg, "pooling_mode", 'gap'))
    pretrained=kwargs.get("pretrained", getattr(cfg, "pretrained", True))

    if 'vit' in backbone_type:
        if 'deit' in backbone_type:
            model = deit_small_patch_features(pretrained=pretrained, **kwargs)
        
        elif 'dinov3' in backbone_type or 'dino' in backbone_type:
            model = dinov3_patch_features()
        
        backbone = ViTBackbone(
            features=model,
            img_size=img_size,
            output_dim=output_dim,
            pooling_mode=pooling_mode,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")
    
    return backbone


# ============================================
# Recommended configurations for HCA-ProtoNet
# ============================================

# RECOMMENDED_CONFIGS = {
#     'default': {
#         'backbone_type': 'vit_deit_base',
#         'pooling_mode': 'gap',  # Best for prototype learning
#         'output_dim': 512,
#         'use_projection': True,
#         'normalize_features': True,
#         'multi_scale': False
#     },
    
#     'high_performance': {
#         'backbone_type': 'vit_dinov2_base',
#         'pooling_mode': 'hybrid',  # CLS + GAP
#         'output_dim': 768,
#         'use_projection': True,
#         'normalize_features': True,
#         'multi_scale': True  # Multi-scale features
#     },
    
#     'lightweight': {
#         'backbone_type': 'vit_deit_small',
#         'pooling_mode': 'gap',
#         'output_dim': 384,
#         'use_projection': True,
#         'normalize_features': True,
#         'multi_scale': False
#     },
    
#     'medical_imaging': {
#         # Optimized for medical imaging (like prostate US)
#         'backbone_type': 'vit_dinov2_base',  # DINOv2 works well on medical
#         'pooling_mode': 'attention',  # Learnable attention
#         'output_dim': 512,
#         'use_projection': True,
#         'normalize_features': True,
#         'multi_scale': True  # Capture multi-scale pathology
#     }
# }


# def get_recommended_config(config_name='default'):
#     """Get a recommended configuration"""
#     return RECOMMENDED_CONFIGS.get(config_name, RECOMMENDED_CONFIGS['default'])


# ============================================
# Example usage
# ============================================
