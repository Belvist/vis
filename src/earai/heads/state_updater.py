"""GRU-like state updater for predictive visual state"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class VisualState:
    """Persistent visual state"""
    scene_tokens: torch.Tensor      # [B, num_scene_tokens, D]
    region_tokens: torch.Tensor     # [B, num_region_tokens, D] (variable)
    region_bboxes: torch.Tensor     # [B, num_region_tokens, 4] normalized
    entity_tracks: dict             # Tracked entities with IDs
    uncertainty: torch.Tensor       # [B, num_tokens] or scalar
    frame_id: int
    timestamp: float


class GatedStateUpdater(nn.Module):
    """
    GRU-like gated state update for visual tokens.
    Lightweight: ~100-200K params.
    
    Inputs:
    - predicted_state: warped previous state [B, N, D]
    - delta_features: encoded residual regions [B, M, D] (M <= N)
    - delta_bboxes: locations of delta features [B, M, 4]
    
    Output: updated state [B, N, D]
    """
    
    def __init__(self, 
                 state_dim: int = 256,
                 hidden_dim: int = 512,
                 num_heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.state_dim = state_dim
        
        # Cross-attention: state tokens attend to delta features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=state_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # GRU-style gates
        self.gate_z = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Sigmoid()
        )
        
        self.gate_r = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Sigmoid()
        )
        
        self.candidate = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        self.norm1 = nn.LayerNorm(state_dim)
        self.norm2 = nn.LayerNorm(state_dim)
        
        # Output projection
        self.out_proj = nn.Linear(state_dim, state_dim)
    
    def forward(self, 
                predicted_state: torch.Tensor,      # [B, N, D]
                delta_features: torch.Tensor,       # [B, M, D]
                delta_bboxes: torch.Tensor,         # [B, M, 4]
                state_bboxes: torch.Tensor) -> torch.Tensor:  # [B, N, 4]
        """
        Update predicted state with delta features.
        Uses spatial alignment (bbox overlap) to route deltas to relevant state tokens.
        """
        B, N, D = predicted_state.shape
        B, M, D = delta_features.shape
        
        # 1. Spatial routing: compute which state tokens each delta affects
        # Simple IoU-based routing
        routing = self._compute_routing(state_bboxes, delta_bboxes)  # [B, N, M]
        
        # 2. Aggregate deltas per state token (weighted by routing)
        routed_deltas = torch.bmm(routing, delta_features)  # [B, N, D]
        
        # 3. Cross-attention: state attends to routed deltas
        state_norm = self.norm1(predicted_state)
        delta_norm = self.norm1(routed_deltas)
        
        attended, _ = self.cross_attn(
            query=state_norm,
            key=delta_norm,
            value=delta_norm
        )
        attended = attended + predicted_state  # residual
        
        # 4. GRU-style gated update
        combined = torch.cat([predicted_state, attended], dim=-1)  # [B, N, 2D]
        
        z = self.gate_z(combined)  # update gate
        r = self.gate_r(combined)  # reset gate
        
        candidate_input = torch.cat([predicted_state, r * attended], dim=-1)
        h_candidate = self.candidate(candidate_input)
        
        # New state
        new_state = (1 - z) * predicted_state + z * h_candidate
        new_state = self.out_proj(new_state)
        
        return new_state
    
    def _compute_routing(self, state_bboxes: torch.Tensor, delta_bboxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU-based routing weights [B, N, M]"""
        B, N, _ = state_bboxes.shape
        B, M, _ = delta_bboxes.shape
        
        # Expand for broadcasting
        s = state_bboxes.unsqueeze(2).expand(-1, -1, M, -1)  # [B, N, M, 4]
        d = delta_bboxes.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, M, 4]
        
        # IoU
        xi1 = torch.max(s[..., 0], d[..., 0])
        yi1 = torch.max(s[..., 1], d[..., 1])
        xi2 = torch.min(s[..., 2], d[..., 2])
        yi2 = torch.min(s[..., 3], d[..., 3])
        
        inter = torch.clamp(xi2 - xi1, min=0) * torch.clamp(yi2 - yi1, min=0)
        area_s = (s[..., 2] - s[..., 0]) * (s[..., 3] - s[..., 1])
        area_d = (d[..., 2] - d[..., 0]) * (d[..., 3] - d[..., 1])
        union = area_s + area_d - inter + 1e-6
        
        iou = inter / union  # [B, N, M]
        
        # Normalize per state token
        routing = iou / (iou.sum(dim=-1, keepdim=True) + 1e-6)
        
        return routing


class UncertaintyEstimator(nn.Module):
    """
    Estimate uncertainty of state tokens.
    High uncertainty -> need correction/keyframe.
    """
    
    def __init__(self, state_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.estimator = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        # Initialize to output low uncertainty (bias = -3 -> sigmoid(-3) ≈ 0.047)
        nn.init.constant_(self.estimator[2].bias, -3.0)
    
    def forward(self, state_tokens: torch.Tensor) -> torch.Tensor:
        """
        state_tokens: [B, N, D]
        Returns: uncertainty [B, N] in [0, 1]
        """
        return self.estimator(state_tokens).squeeze(-1)


class KeyframeDecider(nn.Module):
    """
    Decide: REUSE / ROI_CORRECT / KEYFRAME
    Based on global uncertainty, residual magnitude, and budget.
    """
    
    def __init__(self, 
                 state_dim: int = 256,
                 hidden_dim: int = 128,
                 reuse_threshold: float = 0.3,  # Higher threshold since uncertainty estimator is untrained
                 roi_threshold: float = 0.5,
                 max_roi_ratio: float = 0.3):
        super().__init__()
        self.reuse_threshold = reuse_threshold
        self.roi_threshold = roi_threshold
        self.max_roi_ratio = max_roi_ratio
        
        self.decider = nn.Sequential(
            nn.Linear(state_dim * 2 + 2, hidden_dim),  # mean_state + max_uncertainty + residual_stats
            nn.GELU(),
            nn.Linear(hidden_dim, 3),  # REUSE, ROI, KEYFRAME logits
        )
    
    def forward(self, 
                state_tokens: torch.Tensor,       # [B, N, D]
                uncertainties: torch.Tensor,      # [B, N]
                residual_magnitude: float,        # scalar or [B]
                roi_area_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (decision_logits [B, 3], decision_probs [B, 3])
        """
        B, N, D = state_tokens.shape
        
        # Global stats
        mean_state = state_tokens.mean(dim=1)  # [B, D]
        max_uncertainty = uncertainties.max(dim=1).values  # [B]
        
        # Residual stats (broadcast if scalar)
        if isinstance(residual_magnitude, float):
            res_mag = torch.full((B, 1), residual_magnitude, device=state_tokens.device)
            roi_ratio = torch.full((B, 1), roi_area_ratio, device=state_tokens.device)
        else:
            res_mag = residual_magnitude.unsqueeze(-1) if residual_magnitude.dim() == 1 else residual_magnitude
            roi_ratio = roi_area_ratio.unsqueeze(-1) if roi_area_ratio.dim() == 1 else roi_area_ratio
        
        features = torch.cat([mean_state, max_uncertainty.unsqueeze(-1), res_mag, roi_ratio], dim=-1)
        
        logits = self.decider(features)
        probs = F.softmax(logits, dim=-1)
        
        return logits, probs
    
    def decide_threshold(self, 
                         uncertainties: torch.Tensor,
                         residual_magnitude: float,
                         roi_area_ratio: float) -> str:
        """Simple threshold-based decision (no learning)"""
        max_unc = uncertainties.max().item()
        
        if max_unc < self.reuse_threshold and residual_magnitude < 0.01 and roi_area_ratio < 0.01:
            return "REUSE"
        elif roi_area_ratio > self.max_roi_ratio or max_unc > self.roi_threshold:
            return "KEYFRAME"
        else:
            return "ROI_CORRECT"