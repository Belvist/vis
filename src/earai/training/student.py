"""Training wrapper for EarAI - makes it a proper nn.Module for training"""
import torch
import torch.nn as nn
from typing import Optional, Tuple

from earai.core.earai import EarAIConfig, DEFAULT_CONFIG
from earai.models.backbone import create_backbone
from earai.heads.adaptive_tokens import create_adaptive_pooler
from earai.heads.motion_compensation import MotionEstimator
from earai.heads.residual_encoder import ResidualEncoder, ROIExtractor, ResidualROI, warp_frame, compute_residual_frame
from earai.heads.state_updater import GatedStateUpdater, UncertaintyEstimator, KeyframeDecider, VisualState
from earai.memory.scene_memory import SceneMemory
from earai.heads.foveated_vision import FoveatedVision


class EarAIStudent(nn.Module):
    """
    EarAI as a trainable nn.Module.
    Wraps all components for end-to-end training.
    """
    
    def __init__(self, config: EarAIConfig = None):
        super().__init__()
        self.config = config or DEFAULT_CONFIG
        
        # Core components - all trainable
        self.backbone = create_backbone(self.config)
        
        # Adaptive token pooler
        self.token_pooler = create_adaptive_pooler({
            "type": "multiscale_tokenlearner",
            "channels_list": [self.config.feature_dim] * 4,
            "num_tokens_per_scale": [4, 4, 4, 4],  # 16 total
            "bottleneck_dim": 64
        })
        
        # Motion compensation (not trained, but used)
        self.motion_estimator = MotionEstimator(
            downsample=4,
            max_features=200,
            min_features=10
        )
        
        # Residual encoder
        self.residual_encoder = ResidualEncoder(
            in_channels=9,
            feature_dim=self.config.feature_dim
        )
        
        # ROI extractor
        self.roi_extractor = ROIExtractor(
            base_resolution=self.config.peripheral_resolution,
            roi_size=(64, 64),
            context_margin=0.1,
            max_rois=8
        )
        
        # State updater
        self.state_updater = GatedStateUpdater(
            state_dim=self.config.feature_dim,
            hidden_dim=512
        )
        
        # Uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator(
            state_dim=self.config.feature_dim
        )
        
        # Keyframe decider
        self.keyframe_decider = KeyframeDecider(
            state_dim=self.config.feature_dim,
            hidden_dim=128
        )
        
        # Scene memory
        self.scene_memory = SceneMemory(self.config)
        
        # Foveated vision
        self.foveated_vision = FoveatedVision(self.config)
        
        # Initialize state
        self.register_buffer('prev_frame', torch.zeros(1, 3, 224, 224))
        self.register_buffer('prev_frame_warped', torch.zeros(1, 3, 224, 224))
        self.prev_state = None
        self.frame_id = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass for a single frame (keyframe path).
        Returns visual tokens: [B, 16, 256]
        """
        B = x.shape[0]
        
        # Full backbone - returns dict with F4, F8, F16, F32
        backbone_out = self.backbone(x)
        
        # Extract multi-scale features in correct order for TokenLearner
        features = [
            backbone_out["F4"],
            backbone_out["F8"],
            backbone_out["F16"],
            backbone_out["F32"],
        ]
        
        # Token learner
        token_result = self.token_pooler(features, return_attention=False)
        # Handle both dict and tensor returns
        if isinstance(token_result, dict):
            tokens = token_result['tokens']
        else:
            tokens = token_result  # [B, 16, 256]
        
        return tokens
    
    def forward_streaming(self, x: torch.Tensor) -> dict:
        """
        Streaming forward with decision logic.
        Returns tokens + decision info.
        """
        B = x.shape[0]
        device = x.device
        
        if self.frame_id == 0:
            # First frame - full keyframe
            tokens = self.forward(x)
            decision = 'KEYFRAME'
        else:
            # Motion compensation - convert tensors to numpy for CV2
            prev_frame_np = self.prev_frame[0].permute(1, 2, 0).detach().cpu().numpy()
            curr_frame_np = x[0].permute(1, 2, 0).detach().cpu().numpy()
            
            import cv2
            prev_gray = cv2.cvtColor((prev_frame_np * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor((curr_frame_np * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
            
            motion = self.motion_estimator.estimate(prev_gray, curr_gray)
            
            if motion is not None:
                # Warp previous frame
                warped, valid_mask = warp_frame(self.prev_frame, motion.matrix)
                
                # Compute residual
                curr_np = x[0].permute(1, 2, 0).detach().cpu().numpy()
                warped_np = warped[0].permute(1, 2, 0).detach().cpu().numpy()
                residual_map, binary_mask = compute_residual_frame(curr_np, warped_np)
                
                # Keyframe decision
                change_ratio = binary_mask.mean()
                if change_ratio > 0.1:
                    decision = 'KEYFRAME'
                    tokens = self.forward(x)
                elif change_ratio > 0.001:
                    decision = 'ROI_CORRECT'
                    # Extract ROIs and encode
                    rois = self.roi_extractor.extract_rois(
                        frame=curr_np,
                        motion_transform=motion,
                        prev_frame=prev_frame_np,
                        change_map=binary_mask,
                        uncertainty_map=None,
                        fovea_requests=[],
                        text_regions=[]
                    )
                    if len(rois) > 0:
                        roi_tokens = self._encode_rois(x, rois)
                        tokens = self._update_state(roi_tokens, rois, x)
                    else:
                        tokens = self.prev_state.scene_tokens if self.prev_state else self.forward(x)
                        decision = 'REUSE'
                else:
                    decision = 'REUSE'
                    tokens = self.prev_state.scene_tokens if self.prev_state else self.forward(x)
            else:
                # No motion estimation - fallback to keyframe
                decision = 'KEYFRAME'
                tokens = self.forward(x)
        
        # Update state
        self.prev_frame = x.detach().clone()
        self.frame_id += 1
        
        return {
            'tokens': tokens,
            'decision': decision,
            'frame_id': self.frame_id
        }
    
    def _encode_rois(self, x: torch.Tensor, rois: list) -> torch.Tensor:
        """Encode ROIs using residual encoder"""
        # Convert ROIs to patches and encode
        roi_patches = []
        for roi in rois:
            # Extract patch from current frame
            h, w = x.shape[-2:]
            x1, y1, x2, y2 = roi.bbox
            patch = x[0, :, int(y1*h):int(y2*h), int(x1*w):int(x2*w)]
            # Resize to 64x64
            import cv2
            patch_np = patch.permute(1, 2, 0).cpu().numpy()
            patch_resized = cv2.resize(patch_np, (64, 64))
            roi_patches.append(torch.from_numpy(patch_resized).permute(2, 0, 1))
        
        if roi_patches:
            roi_batch = torch.stack(roi_patches).to(x.device).float() / 255.0
            roi_features = self.residual_encoder(roi_batch)
            return roi_features
        return torch.zeros(0, self.config.feature_dim, device=x.device)
    
    def _update_state(self, roi_tokens: torch.Tensor, rois: list, x: torch.Tensor) -> torch.Tensor:
        """Update state with ROI corrections"""
        if self.prev_state is None:
            return self.forward(x)
        
        prev_tokens = self.prev_state.scene_tokens.unsqueeze(0)
        prev_centroids = self.prev_state.token_centroids.unsqueeze(0)
        
        # Convert ROIs to bbox tensor
        roi_bboxes = torch.tensor([r.bbox for r in rois], device=x.device).unsqueeze(0).float()
        
        updated, _ = self.state_updater(
            prev_tokens=prev_tokens,
            prev_centroids=prev_centroids,
            correction_features=roi_tokens.unsqueeze(0),
            roi_bboxes=roi_bboxes,
            uncertainty=self.prev_state.uncertainty.unsqueeze(0) if self.prev_state and self.prev_state.uncertainty is not None else None
        )
        
        return updated.squeeze(0)
    
    def reset(self):
        """Reset streaming state"""
        self.prev_frame.zero_()
        self.prev_frame_warped.zero_()
        self.prev_state = None
        self.frame_id = 0
        self.motion_estimator.reset()


def create_student_model(config: dict) -> EarAIStudent:
    """Factory for trainable EarAI student"""
    earai_config = EarAIConfig(
        backbone_pretrained=config.get('backbone_pretrained', True),
        feature_dim=config.get('feature_dim', 256),
        num_scene_tokens=config.get('num_scene_tokens', 16),
        peripheral_resolution=config.get('peripheral_resolution', (224, 224)),
        fovea_resolution=config.get('fovea_resolution', (600, 600)),
        max_entities=config.get('max_entities', 100)
    )
    
    return EarAIStudent(earai_config)