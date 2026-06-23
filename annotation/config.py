"""Configuration schema for annotation pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class SamplingConfig:
    """Configuration for frame sampling."""

    mode: Literal["uniform", "subtask_aware"] = "uniform"
    frames_per_episode: int | None = None
    stride_frames: int | None = None
    stride_seconds: float | None = None
    keep_subtask_boundaries: bool = True


@dataclass
class DiscoveryConfig:
    """Configuration for object discovery layer."""

    mode: Literal["instruction", "vocab", "auto"] = "instruction"
    extractor: Literal["qwen", "rule", "mock", "manual"] = "qwen"
    always_include: list[str] = field(default_factory=lambda: ["robot hand"])
    vocab_file: Path | None = None  # Required if mode == "vocab"
    qwen_model_path: Path | None = None  # Deprecated; Qwen now uses HTTP config below
    qwen_endpoint: str = "http://localhost:8000/v1/chat/completions"
    qwen_model: str = "qwen3-placeholder"
    qwen_temperature: float = 0.0
    qwen_max_tokens: int = 128
    qwen_timeout: float = 30.0
    qwen_api_key: str | None = None
    manual_queries: dict[str, list[str]] = field(default_factory=dict)
    instruction_source: Literal["episode_field", "join_file", "none"] = "episode_field"
    instruction_field: str = "expand_task"
    instruction_join_path: Path | None = None
    instruction_join_episode_key: str = "episode_index"
    instruction_join_file_key: str = "episode_index"
    instruction_join_text_field: str = "expand_task"
    default_instruction: str = ""


@dataclass
class SegmentationConfig:
    """Configuration for segmentation layer."""

    model_path: Path | None = None
    model_type: str = "sam3"  # For future extensibility
    confidence_threshold: float = 0.5
    mask_threshold: float = 0.5  # SAM3 mask binarization threshold
    max_instances_per_query: int = 10


@dataclass
class DepthConfig:
    """Configuration for depth estimation layer."""

    model_path: Path | None = None
    model_type: str = "depth_anything_v3"
    output_metric: bool = True  # If False, outputs relative depth
    debug_depth_range: bool = False  # If True, log min/max/mean for debugging
    process_res: int = 504
    process_res_method: str = "upper_bound_resize"
    use_ray_pose: bool = False
    fx: float | None = None
    fy: float | None = None
    calibration_width: int | None = None
    calibration_height: int | None = None


@dataclass
class StorageConfig:
    """Configuration for storage layer."""

    output_dir: Path
    segmentation_output_dir: Path | None = None
    depth_output_dir: Path | None = None
    mask_format: str = "parquet"  # compressed RLE in parquet
    depth_format: str = "png16"  # 16-bit PNG
    overwrite: bool = False  # If True, re-annotate existing frames


@dataclass
class QCConfig:
    """Configuration for quality control visualization."""

    enabled: bool = True
    num_frames_per_episode: int = 5
    output_dir: Path | None = None
    colormap: str = "turbo"  # Colormap for depth visualization


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    dataset_path: Path
    camera_names: list[str]  # e.g., ["observation.images.top"]
    dataset_type: Literal["auto", "lerobot_v3", "mock"] = "auto"
    dry_run: bool = False  # If True, only run discovery layer
    stage: Literal["segmentation", "depth", "both"] = "both"

    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    qc: QCConfig = field(default_factory=QCConfig)

    # Execution control
    checkpoint_dir: Path | None = None  # If None, uses output_dir/.checkpoints
    segmentation_checkpoint_dir: Path | None = None
    depth_checkpoint_dir: Path | None = None
    num_workers: int = 1  # For future parallel processing
    episode_indices: list[int] | None = None
    frame_indices: list[int] | None = None
    sample_frames_per_episode: int | None = None
    max_episodes: int | None = None
    max_frames_per_episode: int | None = None
    log_level: str = "INFO"
