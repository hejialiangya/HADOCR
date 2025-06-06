import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

# You can still use the same Block definition if you wish,
# but we introduce FRMHBlock / FRMVBlock below for clarity.
# from openrec.modeling.common import Block

class FRMHBlock(nn.Module):
    """
    Horizontal Feature Rearrangement Block:
    Interprets each row (H dimension) of the feature map
    and learns to reorder features along the width (W dimension).
    """

    def __init__(self, dim):
        super(FRMHBlock, self).__init__()
        # These are analogous to W^q, W^k, W^v in the paper
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)

        # A small MLP + LN for post-processing
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: (B, C, H, W)
        We'll treat each row as a sequence of length W in dim C.
        Return the rearranged feature map, still (B, C, H, W).
        """
        B, C, H, W = x.shape
        # Reshape to (B*H, W, C)
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H, W, C)

        # Compute queries/keys/values
        q = self.Wq(x_flat)  # (B*H, W, C)
        k = self.Wk(x_flat)  # (B*H, W, C)
        v = self.Wv(x_flat)  # (B*H, W, C)

        # Attention logits: (B*H, W, C) x (B*H, C, W) -> (B*H, W, W)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (C ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)  # row-wise softmax

        # Weighted sum for rearrangement: (B*H, W, W) x (B*H, W, C) -> (B*H, W, C)
        x_rearr = torch.matmul(attn_probs, v)

        # Post-process
        x_rearr = self.ln(x_rearr + x_flat)  # residual + LN
        x_rearr = x_rearr + self.mlp(x_rearr)  # another residual

        # Reshape back to (B, C, H, W)
        x_out = x_rearr.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x_out


class FRMVBlock(nn.Module):
    """
    Vertical Feature Rearrangement Block:
    After rows are rearranged, we reorder across the vertical dimension
    using a single "selector" token (T^s in the paper) for each column feature.
    """

    def __init__(self, dim):
        super(FRMVBlock, self).__init__()
        # Same idea: W^q, W^k, W^v for columns
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.ln = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        # A single "selecting token" T^s in the paper:
        self.select_token = nn.Parameter(torch.zeros(1, 1, dim))
        trunc_normal_(self.select_token, mean=0, std=0.02)

    def forward(self, x):
        """
        x: (B, C, H, W)
        We'll consider columns as sequences of length H in dim C,
        but we also have one "selector token" that attends to each column.
        """
        B, C, H, W = x.shape

        # (B, W, H, C)
        x_perm = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
        # Insert the single selecting token at the front
        # so the sequence has length (H + 1).
        select_token = self.select_token.repeat(B * W, 1, 1)
        x_cat = torch.cat([select_token, x_perm], dim=1)  # (B*W, H+1, C)

        # Compute q/k/v
        q = self.Wq(x_cat)  # (B*W, H+1, C)
        k = self.Wk(x_cat)
        v = self.Wv(x_cat)

        # Self-attention
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (C ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)  # shape (B*W, H+1, H+1)

        x_rearr = torch.matmul(attn_probs, v)  # (B*W, H+1, C)

        # Post-process
        x_rearr = self.ln(x_rearr + x_cat)
        x_rearr = x_rearr + self.mlp(x_rearr)

        # The "selector" row is x_rearr[:,0,:] if we only want one summary,
        # but in the paper's vertical rearrangement, we use the columns 1..H+1.
        # We'll discard the very first token and keep the rest rearranged:
        x_rearr = x_rearr[:, 1:, :]  # shape: (B*W, H, C)

        # Reshape back: (B, W, H, C) -> (B, C, H, W)
        x_rearr = x_rearr.reshape(B, W, H, C).permute(0, 3, 2, 1)
        return x_rearr


class RCTCDecoderFRM(nn.Module):
    """
    A drop-in replacement for the original RCTCDecoder
    that uses two-stage FRM (horizontal then vertical)
    before producing the final logits.
    Input/Output shapes remain the same as the original code.
    """

    def __init__(self,
                 in_channels,
                 out_channels=6625,
                 return_feats=False,
                 **kwargs):
        super(RCTCDecoderFRM, self).__init__()
        self.out_channels = out_channels
        self.return_feats = return_feats

        # First pass: horizontal rearrangement
        self.frm_h = FRMHBlock(in_channels)
        # Second pass: vertical rearrangement
        self.frm_v = FRMVBlock(in_channels)

        # Final linear mapping to vocabulary size
        self.fc = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x, data=None):
        """
        x: (B, C, H, W). Same as original.
        Returns: either (B, W, out_channels) or both (feats, logits).
        """
        B, C, H, W = x.shape

        # 1) Horizontal rearrangement
        x_h = self.frm_h(x)  # shape still (B, C, H, W)

        # 2) Vertical rearrangement
        x_v = self.frm_v(x_h)  # shape (B, C, H, W)

        # Here, we want to produce final per‐column predictions, so
        # we can do a global average or attention across H.  Below we
        # simply flatten along H, then treat W as the "sequence" dimension:
        # shape => (B, C, H*W).  We want (B, W, C) at the end, so let's do:
        feats = x_v.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Just as in your original code, let's interpret the "time" axis as H*W
        # or you could do something like picking out W as the "time" dimension.
        # For simplicity, let's reshape to (B, W, C) if that's your convention:
        if W > 0:
            # Let's group columns back as the "time" axis:
            feats = feats.reshape(B, H, W, C)
            # E.g. take an average across H to get shape (B, W, C):
            feats = feats.mean(dim=1)

        # Classify
        predicts = self.fc(feats)  # (B, W, out_channels)

        if self.return_feats:
            return feats, predicts
        else:
            if not self.training:
                # Same inference behavior: apply softmax
                predicts = F.softmax(predicts, dim=2)
            return predicts
