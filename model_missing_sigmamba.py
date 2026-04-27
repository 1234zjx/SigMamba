import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from mamba_ssm import Mamba
import signatory


# ==========================================
# Multi-Scale TCN
# ==========================================
class DilatedInceptionTCN(nn.Module):
    def __init__(
        self,
        cin=3,
        cout=32,
        kernel_set=[2, 3, 6, 7],
        dilation_factor=2
    ):
        super().__init__()

        branch_out = cout // len(kernel_set)

        self.branches = nn.ModuleList([
            weight_norm(
                nn.Conv2d(
                    cin,
                    branch_out,
                    kernel_size=(1, k),
                    dilation=(1, dilation_factor)
                )
            )
            for k in kernel_set
        ])

    def forward(self, x):
        outs = [b(x) for b in self.branches]

        minT = min(o.size(-1) for o in outs)
        outs = [o[..., -minT:] for o in outs]

        return torch.cat(outs, dim=1)


# ==========================================
# Signature Block
# ==========================================
class SignatureBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        sig_depth=3,
        window_len=8,
        proj_dim=None
    ):
        super().__init__()

        self.window_len = window_len
        self.sig_depth = sig_depth

        sig_channels = signatory.signature_channels(
            input_dim,
            sig_depth
        )

        self.proj = None

        if proj_dim is not None:
            self.proj = nn.Linear(
                sig_channels,
                proj_dim
            )

    def rolling_window(self, x):
        B, T, D = x.shape
        W = self.window_len

        windows = []

        for t in range(T):
            start = max(0, t - W + 1)

            w = x[:, start:t+1]

            if w.shape[1] < W:
                pad = W - w.shape[1]

                w = torch.cat(
                    [w[:, :1].repeat(1, pad, 1), w],
                    dim=1
                )

            windows.append(w)

        return torch.stack(windows, dim=1)

    def forward(self, x):
        B, T, D = x.shape

        windows = self.rolling_window(x)

        windows = windows.reshape(
            B * T,
            self.window_len,
            D
        )

        sig = signatory.signature(
            windows,
            depth=self.sig_depth
        )

        sig = sig.reshape(B, T, -1)

        if self.proj is not None:
            sig = self.proj(sig)

        return sig


# ==========================================
# Cross Variable Attention
# ==========================================
class CrossVariableAttention(nn.Module):
    def __init__(self, num_vars, dim_per_var):
        super().__init__()

        self.num_vars = num_vars
        self.dim_per_var = dim_per_var

        self.attn = nn.MultiheadAttention(
            embed_dim=dim_per_var,
            num_heads=4,
            batch_first=True
        )

    def forward(self, x):
        B, T, D = x.shape

        x = x.view(
            B,
            T,
            self.num_vars,
            self.dim_per_var
        )

        outs = []

        for t in range(T):
            out, _ = self.attn(
                x[:, t],
                x[:, t],
                x[:, t]
            )

            outs.append(out)

        return torch.stack(
            outs,
            dim=1
        ).reshape(B, T, -1)


# ==========================================
# Main Model
# ==========================================
class MissingAwareSigMamba(nn.Module):
    def __init__(
        self,
        num_vars,
        pred_len=96,
        tcn_hidden=32,
        sig_dim=8,
        d_model=128,
        use_signature=True,
        use_cross_attn=True,
        use_mask=True,
        use_delta=True,
    ):
        super().__init__()

        self.num_vars = num_vars
        self.pred_len = pred_len

        self.use_signature = use_signature
        self.use_cross_attn = use_cross_attn
        self.use_mask = use_mask
        self.use_delta = use_delta

        in_channels = (
            1
            + int(use_mask)
            + int(use_delta)
        )

        self.temporal_encoder = DilatedInceptionTCN(
            cin=in_channels,
            cout=tcn_hidden
        )

        self.var_proj = nn.Linear(
            tcn_hidden,
            sig_dim
        )

        self.signature = SignatureBlock(
            sig_dim,
            sig_depth=3,
            window_len=8,
            proj_dim=sig_dim
        )

        self.cross_attn = CrossVariableAttention(
            num_vars,
            sig_dim
        )

        self.fusion_proj = nn.Linear(
            2 * num_vars * sig_dim,
            d_model
        )

        self.mamba = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2
        )

        self.head = nn.Linear(
            d_model,
            pred_len * num_vars
        )

    def forward(self, x, mask, delta):
        B, T, C = x.shape

        feat_list = []

        for i in range(C):

            xi = x[:, :, i:i+1]
            mi = mask[:, :, i:i+1]
            di = delta[:, :, i:i+1]

            feats = [xi]

            if self.use_mask:
                feats.append(mi)

            if self.use_delta:
                feats.append(di)

            inp = torch.cat(
                feats,
                dim=-1
            )

            inp = inp.permute(
                0, 2, 1
            ).unsqueeze(2)

            feat = self.temporal_encoder(inp)

            feat = feat.squeeze(2).permute(
                0, 2, 1
            )

            feat = self.var_proj(feat)

            if self.use_signature:
                feat = self.signature(feat)

            feat_list.append(feat)

        temporal_feat = torch.cat(
            feat_list,
            dim=-1
        )

        if self.use_cross_attn:
            attn_feat = self.cross_attn(
                temporal_feat
            )

            fusion = torch.cat(
                [temporal_feat, attn_feat],
                dim=-1
            )
        else:
            fusion = torch.cat(
                [temporal_feat, temporal_feat],
                dim=-1
            )

        out = self.mamba(
            self.fusion_proj(fusion)
        )

        out = out[:, -1]

        pred = self.head(out)

        pred = pred.view(
            B,
            self.pred_len,
            C
        )

        return pred
