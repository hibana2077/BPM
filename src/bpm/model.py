from __future__ import annotations
import torch
import torch.nn as nn
import timm


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, hidden_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class TimmtBPM(nn.Module):
    """Backbone from timm, exposes embeddings and classifier logits.

    - forward(x) returns logits
    - forward_features(x) returns pooled embedding
    - forward_with_head(x) returns (logits, embedding)
    - forward_projected(x) returns projection for SSL heads
    """
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True, proj_dim: int = 256):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        # Detect feature dimension for projection head
        if hasattr(self.backbone, 'num_features'):
            feat_dim = self.backbone.num_features
        else:
            # fallback: do a dummy forward_features
            dummy = torch.zeros(1, 3, 224, 224)
            with torch.no_grad():
                feats = self.backbone.forward_features(dummy)
                if feats.ndim == 4:
                    feats = feats.mean([2, 3])
                feat_dim = feats.shape[-1]
        self.proj = ProjectionHead(feat_dim, out_dim=proj_dim)

    def forward_features(self, x):
        feats = self.backbone.forward_features(x)
        # Prefer model's own head pooling when available (e.g., ViT/Swin)
        if hasattr(self.backbone, 'forward_head'):
            # pre_logits=True returns pooled features before classifier
            emb = self.backbone.forward_head(feats, pre_logits=True)
            return emb
        # Fallback: pool CNN-style features
        if feats.ndim == 4:
            feats = feats.mean([2, 3])
        elif feats.ndim == 3:
            # (B, N, C) -> global token average
            feats = feats.mean(dim=1)
        return feats

    def forward_projected(self, x):
        z = self.forward_features(x)
        return self.proj(z)

    def forward(self, x):
        return self.backbone(x)

    def forward_with_head(self, x):
        # Compute both logits and pooled embedding in a model-aware way
        if hasattr(self.backbone, 'forward_head'):
            raw = self.backbone.forward_features(x)
            logits = self.backbone.forward_head(raw, pre_logits=False)
            emb = self.backbone.forward_head(raw, pre_logits=True)
            return logits, emb
        # Fallback for backbones without forward_head
        emb = self.forward_features(x)
        logits = self.backbone.get_classifier()(emb) if hasattr(self.backbone, 'get_classifier') else self.backbone(x)
        return logits, emb
