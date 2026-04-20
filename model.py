"""
Slot Attention Models
  - SlotAttention: core iterative attention module (Locatello et al. 2020)
  - SlotAutoencoder: full reconstruction model (vanilla, frame-by-frame)
  - TemporalSlotAutoencoder: adds temporal identity propagation (Eq. 8 from Chung et al.)

Both models share the same encoder/decoder and slot attention core.
The only difference is whether slots are initialized randomly each frame
or carried over from the previous frame.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class SoftPositionEmbed(nn.Module):
    """Learnable soft position embeddings (used in original slot attention paper)."""
    def __init__(self, hidden_size, resolution):
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.register_buffer("grid", self._build_grid(resolution))

    @staticmethod
    def _build_grid(resolution):
        ranges = [torch.linspace(0.0, 1.0, steps=r) for r in resolution]
        grid = torch.meshgrid(*ranges, indexing="ij")
        grid = torch.stack(grid, dim=-1)
        grid = grid.unsqueeze(0)  # (1, H, W, 2)
        return torch.cat([grid, 1.0 - grid], dim=-1)  # (1, H, W, 4)

    def forward(self, x):
        # x: (B, H, W, C)
        pos = self.embedding(self.grid)   # (1, H, W, hidden)
        return x + pos


# ---------------------------------------------------------------------------
# CNN Encoder / Decoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_size=64, resolution=(64, 64)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, 5, padding=2), nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, 5, padding=2), nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, 5, padding=2), nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, 5, padding=2), nn.ReLU(),
        )
        self.pos_embed = SoftPositionEmbed(hidden_size, resolution)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.net(x)                              # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)                   # (B, H, W, C)
        x = self.pos_embed(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)                  # (B, N, C)
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x                                     # (B, N, hidden)


class SpatialBroadcastDecoder(nn.Module):
    """Each slot is broadcast to a grid, then decoded independently."""
    def __init__(self, slot_dim, hidden_size=64, resolution=(64, 64)):
        super().__init__()
        self.resolution = resolution
        self.pos_embed = SoftPositionEmbed(slot_dim, resolution)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(slot_dim, hidden_size, 5, padding=2), nn.ReLU(),
            nn.ConvTranspose2d(hidden_size, hidden_size, 5, padding=2), nn.ReLU(),
            nn.ConvTranspose2d(hidden_size, hidden_size, 5, padding=2), nn.ReLU(),
            nn.ConvTranspose2d(hidden_size, hidden_size, 5, padding=2), nn.ReLU(),
            nn.ConvTranspose2d(hidden_size, 4, 3, padding=1),  # RGB + alpha
        )

    def forward(self, slots):
        # slots: (B, K, slot_dim)
        B, K, D = slots.shape
        H, W = self.resolution

        # Broadcast each slot to spatial grid
        x = slots.reshape(B * K, D, 1, 1).expand(B * K, D, H, W)
        x = x.permute(0, 2, 3, 1)                   # (BK, H, W, D)
        x = self.pos_embed(x)
        x = x.permute(0, 3, 1, 2)                   # (BK, D, H, W)

        x = self.net(x)                              # (BK, 4, H, W)
        x = x.reshape(B, K, 4, H, W)

        rgb   = x[:, :, :3]                          # (B, K, 3, H, W)
        alpha = x[:, :, 3:4]                         # (B, K, 1, H, W)
        return rgb, alpha


# ---------------------------------------------------------------------------
# Slot Attention Module
# ---------------------------------------------------------------------------

class SlotAttention(nn.Module):
    """
    Slot Attention module (Locatello et al., NeurIPS 2020).
    Iteratively binds K slots to N input features via competitive softmax.
    """
    def __init__(self, n_slots, slot_dim, n_iters=3, hidden_dim=128, eps=1e-8):
        super().__init__()
        self.n_slots   = n_slots
        self.slot_dim  = slot_dim
        self.n_iters   = n_iters
        self.eps       = eps
        self.scale     = slot_dim ** -0.5

        # Slot initialization parameters (shared Gaussian)
        self.slot_mu    = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slot_sigma = nn.Parameter(torch.ones(1, 1, slot_dim))

        # Attention projections
        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(slot_dim, slot_dim, bias=False)

        # Slot update
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim * 2), nn.ReLU(),
            nn.Linear(slot_dim * 2, slot_dim),
        )
        self.norm_inputs = nn.LayerNorm(slot_dim)
        self.norm_slots  = nn.LayerNorm(slot_dim)

    def init_slots(self, batch_size, device):
        """Random initialization from learned Gaussian."""
        noise = torch.randn(batch_size, self.n_slots, self.slot_dim, device=device)
        return self.slot_mu + self.slot_sigma.abs() * noise

    def forward(self, inputs, slots=None):
        """
        Args:
            inputs: (B, N, slot_dim) — encoded image features
            slots:  (B, K, slot_dim) or None — if None, randomly initialized
        Returns:
            slots:  (B, K, slot_dim) — final slot representations
            attn:   (B, K, N)        — attention weights (for visualization)
        """
        B, N, D = inputs.shape
        inputs = self.norm_inputs(inputs)

        k = self.to_k(inputs)   # (B, N, D)
        v = self.to_v(inputs)   # (B, N, D)

        if slots is None:
            slots = self.init_slots(B, inputs.device)

        attn_weights = None
        for _ in range(self.n_iters):
            slots_prev = slots
            slots_normed = self.norm_slots(slots)
            q = self.to_q(slots_normed)  # (B, K, D)

            # Attention logits: (B, N, K)
            dots = torch.einsum("bnd,bkd->bnk", k, q) * self.scale

            # Softmax over SLOTS — the competitive step
            attn = dots.softmax(dim=2)              # (B, N, K)
            attn_weights = attn.permute(0, 2, 1)    # (B, K, N) for visualization

            # Normalize for weighted mean
            attn_sum = attn.sum(dim=1, keepdim=True) + self.eps
            attn_norm = attn / attn_sum             # (B, N, K)

            # Weighted mean of values per slot
            updates = torch.einsum("bnk,bnd->bkd", attn_norm, v)  # (B, K, D)

            # GRU update
            slots = self.gru(
                updates.reshape(B * self.n_slots, D),
                slots_prev.reshape(B * self.n_slots, D),
            ).reshape(B, self.n_slots, D)

            slots = slots + self.mlp(slots)

        return slots, attn_weights


# ---------------------------------------------------------------------------
# Full Autoencoder Models
# ---------------------------------------------------------------------------

class SlotAutoencoder(nn.Module):
    """
    Vanilla Slot Attention autoencoder.
    Processes each frame independently — no temporal memory.
    This is the BASELINE that exhibits binding drift under occlusion.
    """
    def __init__(self, resolution=(64, 64), n_slots=4, slot_dim=64,
                 encoder_hidden=64, n_iters=3):
        super().__init__()
        self.n_slots  = n_slots
        self.slot_dim = slot_dim

        self.encoder      = Encoder(3, encoder_hidden, resolution)
        self.encoder_proj = nn.Sequential(
            nn.LayerNorm(encoder_hidden),
            nn.Linear(encoder_hidden, slot_dim),
        )
        self.slot_attention = SlotAttention(n_slots, slot_dim, n_iters)
        self.decoder        = SpatialBroadcastDecoder(slot_dim, encoder_hidden, resolution)

    def forward(self, x, prev_slots=None):
        """
        Args:
            x:          (B, 3, H, W)
            prev_slots: ignored in vanilla version
        Returns:
            recon:      (B, 3, H, W)
            masks:      (B, K, 1, H, W) softmax alpha masks
            slots:      (B, K, slot_dim)
            attn:       (B, K, N)
        """
        feats = self.encoder(x)           # (B, N, encoder_hidden)
        feats = self.encoder_proj(feats)  # (B, N, slot_dim)

        # Always random init — no temporal propagation
        slots, attn = self.slot_attention(feats, slots=None)

        rgb, alpha = self.decoder(slots)  # (B, K, 3/1, H, W)

        # Softmax masks across slots
        masks = alpha.softmax(dim=1)         # (B, K, 1, H, W)
        slot_recons = rgb * masks            # (B, K, 3, H, W) per-slot reconstruction
        recon = slot_recons.sum(dim=1)       # (B, 3, H, W) combined

        return recon, masks, slots, attn, slot_recons


class TemporalSlotAutoencoder(nn.Module):
    """
    Temporal Slot Autoencoder — adds temporal identity propagation.

    Key change: slots are initialized from the PREVIOUS FRAME's output
    instead of randomly. This implements Equation 8 from Chung et al.:

        s_t^(0) = RandomInit()      if t == 0
        s_t^(0) = s_{t-1}^(T)      if t > 0

    This gives each slot a strong prior toward the object it was tracking,
    which helps maintain binding stability through occlusion events.
    """
    def __init__(self, resolution=(64, 64), n_slots=4, slot_dim=64,
                 encoder_hidden=64, n_iters=3):
        super().__init__()
        self.n_slots  = n_slots
        self.slot_dim = slot_dim

        self.encoder      = Encoder(3, encoder_hidden, resolution)
        self.encoder_proj = nn.Sequential(
            nn.LayerNorm(encoder_hidden),
            nn.Linear(encoder_hidden, slot_dim),
        )
        self.slot_attention = SlotAttention(n_slots, slot_dim, n_iters)
        self.decoder        = SpatialBroadcastDecoder(slot_dim, encoder_hidden, resolution)

    def forward(self, x, prev_slots=None):
        """
        Args:
            x:          (B, 3, H, W)
            prev_slots: (B, K, slot_dim) or None — previous frame's slots
        Returns:
            recon:      (B, 3, H, W)
            masks:      (B, K, 1, H, W)
            slots:      (B, K, slot_dim)  ← pass this back as prev_slots next frame
            attn:       (B, K, N)
        """
        feats = self.encoder(x)
        feats = self.encoder_proj(feats)

        # THE KEY DIFFERENCE: use prev_slots as initialization if available
        slots, attn = self.slot_attention(feats, slots=prev_slots)

        rgb, alpha = self.decoder(slots)
        masks = alpha.softmax(dim=1)
        slot_recons = rgb * masks            # (B, K, 3, H, W) per-slot reconstruction
        recon = slot_recons.sum(dim=1)       # (B, 3, H, W) combined

        return recon, masks, slots, attn, slot_recons