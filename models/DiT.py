import torch
import torch.nn as nn
from models.embeddings import SinusoidalPositionEmbeddings


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, hidden_size):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size  = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size, patch_size, out_channels, hidden_size):
        """
        Reassemble (B, N, C*p*p) token sequence → (B, C, H, W) image.
 
        No linear projection here — FinalLayer has already projected from
        hidden_size to C*p*p. This module only does the spatial reshape.
        """
        super().__init__()
        self.patch_size   = patch_size
        self.h_patches    = img_size // patch_size
        self.out_channels = out_channels

    def forward(self, x):
        # x: (B, N, C*p*p)  — already projected by FinalLayer
        B = x.shape[0]
        p, h, C = self.patch_size, self.h_patches, self.out_channels
        x = x.reshape(B, h, h, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)
        return x.reshape(B, C, h * p, h * p)


def modulate(x, shift, scale):
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = nn.MultiheadAttention(hidden_size, num_heads,
                                           dropout=dropout_rate, batch_first=True)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden, hidden_size), nn.Dropout(dropout_rate),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        shift1, scale1, gate1, shift2, scale2, gate2 = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        x_mod = modulate(self.norm1(x), shift1, scale1)
        attn_out, _ = self.attn(x_mod, x_mod, x_mod, need_weights=False)
        x = x + gate1.unsqueeze(1) * attn_out
        x_mod = modulate(self.norm2(x), shift2, scale2)
        x = x + gate2.unsqueeze(1) * self.mlp(x_mod)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm   = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class DiT(nn.Module):
    """
    Diffusion Transformer — drop-in replacement for UNet.

    Keeps the same forward(x, t) signature and reuses
    SinusoidalPositionEmbeddings from models/embeddings.py,
    so t is an integer index tensor in [0, total_time_steps).

    Constructor parameter names deliberately mirror UNet where they overlap
    (dropout_rate, time_multiple, total_time_steps, time_emb_max_frec)
    so the same YAML config section drives both backbones.
    """
    def __init__(
        self,
        input_channels:    int   = 3,
        output_channels:   int   = 3,
        img_size:          int   = 64,
        patch_size:        int   = 4,
        hidden_size:       int   = 256,
        depth:             int   = 6,
        num_heads:         int   = 4,
        mlp_ratio:         float = 4.0,
        dropout_rate:      float = 0.1,
        time_multiple:     int   = 4,        # same name as UNet
        total_time_steps:  int   = 1000,     # same name as UNet
        time_emb_max_frec: float = 10000.0,  # same name as UNet
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.input_channels  = input_channels
        self.output_channels = output_channels

        # time_emb_dims_exp uses the same formula as UNet
        time_emb_dims_exp = hidden_size * time_multiple

        # ── Timestep embedding — identical to UNet ────────────────────────
        # t : (B,) long  →  (B, time_emb_dims_exp)
        self.time_embeddings = SinusoidalPositionEmbeddings(
            total_time_steps  = total_time_steps,
            time_emb_dims     = hidden_size,
            time_emb_dims_exp = time_emb_dims_exp,
            time_emb_max_frec = time_emb_max_frec,
        )
        # Project from time_emb_dims_exp → hidden_size for AdaLN conditioning
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dims_exp, hidden_size),
            nn.SiLU(),
        )

        # ── Patch embedding ───────────────────────────────────────────────
        self.patch_embed = PatchEmbed(img_size, patch_size, input_channels, hidden_size)
        num_patches = self.patch_embed.num_patches

        # ── Learned spatial positional encoding for the N patch tokens ────
        # (separate from the temporal sinusoidal encoding above)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── Transformer blocks ────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, dropout_rate)
            for _ in range(depth)
        ])

        # ── Output ────────────────────────────────────────────────────────
        self.final_layer = FinalLayer(hidden_size, patch_size, output_channels)
        self.unpatch     = PatchUnEmbed(img_size, patch_size, output_channels, hidden_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.view(m.weight.size(0), -1))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, H, W)  noisy image
            t : (B,)  long    integer timestep in [0, total_time_steps)
                              — same convention as UNet.forward
        Returns:
            (B, C, H, W)  predicted noise (DDPM) or velocity field (FM)
        """
        # 1. Timestep conditioning  →  (B, hidden_size)
        time_emb = self.time_embeddings(t)   # (B, time_emb_dims_exp)
        c = self.time_proj(time_emb)         # (B, hidden_size)

        # 2. Patchify + spatial positional encoding
        tokens = self.patch_embed(x) + self.pos_embed   # (B, N, D)

        # 3. DiT blocks
        for block in self.blocks:
            tokens = block(tokens, c)

        # 4. Project back to pixel space
        tokens = self.final_layer(tokens, c)  # (B, N, p²·C_out)
        return self.unpatch(tokens)           # (B, C_out, H, W)