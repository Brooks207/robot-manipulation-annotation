"""Depth Anything V3 depth estimation implementation."""

import logging
from pathlib import Path

import numpy as np

from ..types import DepthResult
from .base import DepthEstimator

logger = logging.getLogger(__name__)


class DepthAnything3Estimator(DepthEstimator):
    """
    Depth estimation using Depth Anything V3.

    Uses standalone depth-anything-3 package (not transformers).
    """

    def __init__(self, model_path: Path | None = None, config: dict | None = None):
        """
        Initialize Depth Anything 3 estimator.

        Args:
            model_path: Path to Depth Anything 3 config file
            config: Depth config dict
        """
        self.model_path = model_path
        self.config = config or {}
        self.model = None
        self.device = None

        if model_path:
            self._load_model()
        else:
            logger.warning(
                "Depth Anything 3 model path not provided. "
                "Estimator not initialized."
            )

    def _load_model(self) -> None:
        """
        Load Depth Anything 3 model.

        Uses depth_anything_3 high-level API.
        """
        logger.info(f"Loading Depth Anything 3 model from {self.model_path}")

        try:
            from depth_anything_3.api import DepthAnything3
            import torch

            # Determine device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")

            # DepthAnything3(model_name=...) only builds the architecture. Use
            # from_pretrained so the official checkpoint weights are loaded.
            self.model = DepthAnything3.from_pretrained(str(self.model_path))
            self.model = self.model.to(self.device)
            self.model.device = self.device
            self.model.eval()

            first_param = next(self.model.parameters()).detach()
            nonzero = int((first_param != 0).sum().item())
            total = first_param.numel()
            logger.info(
                "DA3 checkpoint parameter check: first_param_nonzero=%d/%d",
                nonzero,
                total,
            )
            if nonzero == 0:
                raise RuntimeError(
                    "DA3 checkpoint appears unloaded: first parameter is all zeros"
                )

            logger.info("Depth Anything 3 model loaded successfully")

        except ImportError as e:
            logger.error(f"Failed to import depth_anything_3: {e}")
            logger.error("Ensure depth-anything-3 package is installed")
            raise
        except Exception as e:
            logger.error(f"Failed to load Depth Anything 3 model: {e}")
            raise

    def estimate_depth(self, frame: np.ndarray, config: dict) -> DepthResult | None:
        """
        Estimate depth for a single frame.

        Args:
            frame: RGB image, shape (H, W, 3), dtype uint8
            config: Depth config with output_metric and debug_depth_range

        Returns:
            DepthResult or None on failure
        """
        if self.model is None:
            logger.error("Depth Anything 3 model not loaded, cannot estimate depth")
            return None

        try:
            return self._estimate_with_da3(frame, config)
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}", exc_info=True)
            return None

    def _estimate_with_da3(
        self, frame: np.ndarray, config: dict
    ) -> DepthResult | None:
        """
        Internal depth estimation logic.

        Converts DA3Metric raw network output to metric depth using focal/300.
        """
        import torch

        process_res = int(config.get("process_res", 504))
        process_res_method = config.get("process_res_method", "upper_bound_resize")
        use_ray_pose = bool(config.get("use_ray_pose", False))

        # Use DA3's high-level inference API
        # Input: list of images (numpy arrays)
        # Output: Prediction object with raw .depth for DA3Metric
        with torch.inference_mode():
            result = self.model.inference(
                [frame],
                process_res=process_res,
                process_res_method=process_res_method,
                use_ray_pose=use_ray_pose,
            )

        # Extract depth map (shape: (1, H, W))
        raw_depth = result.depth[0].astype(np.float32)  # Get first image's depth

        # Get config parameters
        output_metric = config.get("output_metric", True)
        debug_depth_range = config.get("debug_depth_range", False)

        if output_metric:
            focal = self._scaled_focal_for_depth(raw_depth.shape, frame.shape, config)
            depth = (raw_depth * (focal / 300.0)).astype(np.float32)
            depth_type = "metric"
            scale = 1.0
        else:
            depth_min = float(raw_depth.min())
            depth_max = float(raw_depth.max())
            depth = ((raw_depth - depth_min) / (depth_max - depth_min + 1e-8)).astype(
                np.float32
            )
            depth_type = "relative"
            scale = float(depth_max - depth_min)

        raw_min = float(raw_depth.min())
        raw_max = float(raw_depth.max())
        raw_mean = float(raw_depth.mean())
        raw_std = float(raw_depth.std())
        depth_min = float(depth.min())
        depth_max = float(depth.max())
        depth_mean = float(depth.mean())
        depth_std = float(depth.std())

        if debug_depth_range:
            logger.info(
                "DA3 raw depth: shape=%s min=%.4f max=%.4f mean=%.4f std=%.4f",
                raw_depth.shape,
                raw_min,
                raw_max,
                raw_mean,
                raw_std,
            )
            logger.info(
                "DA3 output depth: type=%s min=%.4f max=%.4f mean=%.4f std=%.4f",
                depth_type,
                depth_min,
                depth_max,
                depth_mean,
                depth_std,
            )

        return DepthResult(
            depth=depth.astype(np.float32),
            depth_type=depth_type,
            scale=scale,
        )

    def _scaled_focal_for_depth(
        self, depth_shape: tuple[int, int], frame_shape: tuple[int, ...], config: dict
    ) -> float:
        fx = config.get("fx")
        fy = config.get("fy")
        calibration_width = config.get("calibration_width")
        calibration_height = config.get("calibration_height")

        if fx is None or fy is None:
            raise ValueError("Metric DA3 depth requires depth.fx and depth.fy in pixels")
        if calibration_width is None or calibration_height is None:
            raise ValueError(
                "Metric DA3 depth requires calibration_width and calibration_height"
            )

        depth_h, depth_w = depth_shape
        scaled_fx = float(fx) * (depth_w / float(calibration_width))
        scaled_fy = float(fy) * (depth_h / float(calibration_height))
        focal = (scaled_fx + scaled_fy) / 2.0

        logger.debug(
            "DA3 metric focal scaling: frame_shape=%s depth_shape=%s "
            "fx=%.6f fy=%.6f focal=%.6f",
            frame_shape,
            depth_shape,
            scaled_fx,
            scaled_fy,
            focal,
        )
        return focal
