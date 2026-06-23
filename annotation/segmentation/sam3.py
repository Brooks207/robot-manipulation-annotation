"""SAM 3 segmentation implementation."""

import logging
from pathlib import Path

import numpy as np

from ..types import InstanceMask
from .base import Segmenter

logger = logging.getLogger(__name__)


class SAM3Segmenter(Segmenter):
    """
    Segmentation using SAM 3 with text/referring prompts.

    Uses HuggingFace transformers SAM3Model and Sam3Processor.
    """

    def __init__(self, model_path: Path | None = None, config: dict | None = None):
        """
        Initialize SAM 3 segmenter.

        Args:
            model_path: Path to SAM 3 model checkpoint or HF model ID
            config: Segmentation config dict
        """
        self.model_path = model_path
        self.config = config or {}
        self.model = None
        self.processor = None
        self.device = None
        self.vision_features_cache = {}  # Cache vision features per frame

        if model_path:
            self._load_model()
        else:
            logger.warning(
                f"SAM 3 model path not provided. "
                "Segmenter not initialized."
            )

    def _load_model(self) -> None:
        """
        Load SAM 3 model from HuggingFace.

        Uses transformers v5.x Sam3Model and Sam3Processor.
        """
        logger.info(f"Loading SAM 3 model from {self.model_path}")

        try:
            from transformers import Sam3Model, Sam3Processor
            import torch

            # Determine device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")

            # Load processor and model
            # model_path can be HF model ID (e.g., "facebook/sam3") or local path
            model_id = str(self.model_path)
            self.processor = Sam3Processor.from_pretrained(model_id)
            self.model = Sam3Model.from_pretrained(model_id).to(self.device)
            self.model.eval()

            logger.info("SAM 3 model loaded successfully")

        except ImportError as e:
            logger.error(f"Failed to import SAM 3 dependencies: {e}")
            logger.error("Ensure transformers>=5.0 is installed")
            raise
        except Exception as e:
            logger.error(f"Failed to load SAM 3 model: {e}")
            raise

    def segment_frame(
        self, frame: np.ndarray, queries: list[str], config: dict
    ) -> list[InstanceMask]:
        """
        Segment frame using SAM 3 text prompts.

        Args:
            frame: RGB image, shape (H, W, 3), dtype uint8
            queries: Text queries for objects to segment
            config: Segmentation config

        Returns:
            List of InstanceMask objects (empty list on failure)
        """
        if self.model is None:
            logger.error("SAM 3 model not loaded, cannot segment")
            return []

        if not queries:
            logger.debug("No queries provided, returning empty list")
            return []

        try:
            return self._segment_with_sam3(frame, queries, config)
        except Exception as e:
            logger.error(f"Segmentation failed: {e}", exc_info=True)
            return []

    def _segment_with_sam3(
        self, frame: np.ndarray, queries: list[str], config: dict
    ) -> list[InstanceMask]:
        """
        Internal segmentation logic using SAM 3.

        Optimized to reuse vision features across queries for the same frame.

        Args:
            frame: RGB frame, shape (H, W, 3), uint8
            queries: List of text queries
            config: Segmentation config with threshold/mask_threshold

        Returns:
            List of InstanceMask objects
        """
        import torch

        # Get config parameters
        threshold = config.get("confidence_threshold", 0.5)
        mask_threshold = config.get("mask_threshold", 0.5)
        max_instances_per_query = config.get("max_instances_per_query", 10)

        results = []

        # Compute vision features once per frame (optimization)
        frame_id = id(frame)  # Use object id as cache key
        if frame_id not in self.vision_features_cache:
            # Encode image to get vision features
            inputs = self.processor(images=frame, return_tensors="pt").to(self.device)
            with torch.no_grad():
                vision_outputs = self.model.get_vision_features(
                    pixel_values=inputs["pixel_values"]
                )
            self.vision_features_cache[frame_id] = {
                "vision_outputs": vision_outputs,
                "original_sizes": inputs.get("original_sizes"),
            }
            logger.debug("Computed and cached vision features for frame")

        cached = self.vision_features_cache[frame_id]
        vision_outputs = cached["vision_outputs"]
        original_sizes = cached["original_sizes"]

        # Process each query using cached vision features
        for query in queries:
            try:
                # Prepare text inputs
                inputs = self.processor(
                    images=frame, text=query, return_tensors="pt"
                ).to(self.device)

                # Forward pass with cached vision features
                with torch.no_grad():
                    outputs = self.model(
                        **inputs,
                        vision_outputs=vision_outputs,  # Reuse cached features
                    )

                # Post-process to get masks
                batch_results = self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=threshold,
                    mask_threshold=mask_threshold,
                    target_sizes=original_sizes.tolist(),
                )

                # Extract results for this query
                query_result = batch_results[0]

                # Convert to InstanceMask objects
                masks = query_result.get("masks", [])
                boxes = query_result.get("boxes", [])
                scores = query_result.get("scores", [])

                # Limit instances per query
                num_instances = min(len(masks), max_instances_per_query)

                for i in range(num_instances):
                    mask_binary = masks[i].cpu().numpy().astype(bool)
                    bbox_xyxy = boxes[i].cpu().numpy()  # [x1, y1, x2, y2]
                    score = float(scores[i].cpu().numpy())

                    # Convert bbox from xyxy to xywh
                    x1, y1, x2, y2 = bbox_xyxy
                    bbox_xywh = (
                        int(x1),
                        int(y1),
                        int(x2 - x1),
                        int(y2 - y1),
                    )

                    results.append(
                        InstanceMask(
                            category=query,
                            mask=mask_binary,
                            score=score,
                            bbox=bbox_xywh,
                        )
                    )

                logger.debug(
                    f"Query '{query}': found {num_instances} instances "
                    f"(threshold={threshold})"
                )

            except Exception as e:
                logger.warning(f"Failed to process query '{query}': {e}")
                # Continue to next query (per-query failure isolation)
                continue

        # Clear cache to avoid memory buildup (this frame won't be revisited)
        if frame_id in self.vision_features_cache:
            del self.vision_features_cache[frame_id]

        logger.info(
            f"SAM 3 segmentation complete: {len(results)} instances "
            f"across {len(queries)} queries"
        )

        return results
