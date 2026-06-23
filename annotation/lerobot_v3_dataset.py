"""Minimal LeRobot v3 dataset reader for annotation verification."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

logger = logging.getLogger(__name__)


class LeRobotV3Dataset:
    """Read episode metadata and video frames from a local LeRobot v3 directory."""

    def __init__(
        self,
        dataset_path: Path,
        camera_names: list[str],
        instruction_config: dict[str, Any],
        episode_indices: list[int] | None = None,
        frame_indices: list[int] | None = None,
        sample_frames_per_episode: int | None = None,
        sampling_config: dict[str, Any] | None = None,
        max_episodes: int | None = None,
        max_frames_per_episode: int | None = None,
        load_frames: bool = True,
    ):
        self.dataset_path = Path(dataset_path)
        self.camera_names = camera_names
        self.instruction_config = instruction_config
        self.frame_indices = frame_indices
        self.sample_frames_per_episode = sample_frames_per_episode
        self.sampling_config = sampling_config or {}
        self.max_frames_per_episode = max_frames_per_episode
        self.load_frames = load_frames
        self.join_instructions: dict[Any, str] | None = None
        self.dataset_fps = self._load_dataset_fps()
        self.subtask_info = self._load_subtask_info()
        self._data_metadata_cache: dict[Path, pd.DataFrame] = {}

        episodes_dir = self.dataset_path / "meta" / "episodes"
        episode_files = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
        if not episode_files:
            raise FileNotFoundError(f"No episode parquet files under {episodes_dir}")

        episodes = [pq.read_table(path).to_pandas() for path in episode_files]
        self.episodes = (
            pd.concat(episodes, ignore_index=True) if len(episodes) > 1 else episodes[0]
        )
        self.episodes = self.episodes.sort_values("episode_index").reset_index(drop=True)

        if episode_indices is not None:
            selected = set(episode_indices)
            self.episodes = self.episodes[
                self.episodes["episode_index"].isin(selected)
            ].reset_index(drop=True)
        if max_episodes is not None:
            self.episodes = self.episodes.head(max_episodes).reset_index(drop=True)

        if self.episodes.empty:
            raise ValueError("No episodes selected")

        if instruction_config.get("instruction_source") == "join_file":
            self.join_instructions = self._load_join_instructions()

        logger.info(
            "Loaded LeRobot v3 dataset: path=%s episodes=%d",
            self.dataset_path,
            len(self.episodes),
        )

    def __len__(self) -> int:
        return len(self.episodes)

    def episode_index_at(self, episode_idx: int) -> int:
        return int(self.episodes.iloc[episode_idx]["episode_index"])

    def get_episode(self, episode_idx: int) -> dict[str, Any]:
        row = self.episodes.iloc[episode_idx]
        actual_episode_idx = int(row["episode_index"])
        length = int(row["length"])
        frame_metadata = self._frame_metadata_for_episode(row, length)
        selected_frames = self._selected_frame_indices(row, length, frame_metadata)
        instruction = self._episode_instruction(row)

        frames: dict[str, dict[int, np.ndarray]] = {}
        if self.load_frames:
            for camera_name in self.camera_names:
                frames[camera_name] = self._load_camera_frames(
                    row, camera_name, selected_frames
                )

        return {
            "episode_index": actual_episode_idx,
            "instruction": instruction,
            "num_frames": length,
            "frame_indices": selected_frames,
            "frame_metadata": {
                frame_idx: frame_metadata[frame_idx] for frame_idx in selected_frames
            },
            "frames": frames,
        }

    def _selected_frame_indices(
        self, row: Any, length: int, frame_metadata: dict[int, dict[str, Any]]
    ) -> list[int]:
        if self.frame_indices is not None:
            selected = [idx for idx in self.frame_indices if 0 <= idx < length]
        elif self.sampling_config.get("mode", "uniform") == "subtask_aware":
            selected = self._subtask_aware_frame_indices(row, length, frame_metadata)
        elif self._uniform_frames_per_episode() is not None:
            count = min(int(self._uniform_frames_per_episode()), length)
            selected = np.linspace(0, length - 1, count, dtype=int).tolist()
        elif self.max_frames_per_episode is not None:
            selected = list(range(min(length, self.max_frames_per_episode)))
        else:
            selected = list(range(length))
        if not selected:
            raise ValueError(f"No valid frame indices selected for episode length {length}")
        return sorted(set(selected))

    def _uniform_frames_per_episode(self) -> int | None:
        configured = self.sampling_config.get("frames_per_episode")
        if configured is not None:
            return int(configured)
        return self.sample_frames_per_episode

    def _subtask_aware_frame_indices(
        self, row: Any, length: int, frame_metadata: dict[int, dict[str, Any]]
    ) -> list[int]:
        stride = self._subtask_stride_frames(frame_metadata)
        keep_boundaries = bool(
            self.sampling_config.get("keep_subtask_boundaries", True)
        )
        actual_episode_idx = int(row["episode_index"])

        missing_subtask = [
            frame_idx
            for frame_idx in range(length)
            if frame_metadata[frame_idx].get("subtask_index") is None
        ]
        if missing_subtask:
            raise ValueError(
                "sampling.mode=subtask_aware requires per-frame subtask_index; "
                f"episode {actual_episode_idx} has {len(missing_subtask)} missing values"
            )

        segments: list[tuple[int, int, int]] = []
        segment_start = 0
        current_subtask = int(frame_metadata[0]["subtask_index"])
        for frame_idx in range(1, length):
            subtask_index = int(frame_metadata[frame_idx]["subtask_index"])
            if subtask_index != current_subtask:
                segments.append((current_subtask, segment_start, frame_idx - 1))
                segment_start = frame_idx
                current_subtask = subtask_index
        segments.append((current_subtask, segment_start, length - 1))

        selected: set[int] = set()
        segment_samples: list[tuple[int, int, int, list[int]]] = []
        for subtask_index, start, end in segments:
            frames = list(range(start, end + 1, stride))
            if keep_boundaries:
                frames.extend([start, end])
            frames = sorted(set(frames))
            selected.update(frames)
            segment_samples.append((subtask_index, start, end, frames))

        selected_frames = sorted(selected)
        logger.info(
            "Episode %d: subtask-aware sampling selected %d/%d frames "
            "(stride=%d frames, keep_boundaries=%s)",
            actual_episode_idx,
            len(selected_frames),
            length,
            stride,
            keep_boundaries,
        )
        for subtask_index, start, end, frames in segment_samples:
            subtask_text = self.subtask_info.get(subtask_index, {}).get("subtask", "")
            label = f" ({subtask_text})" if subtask_text else ""
            logger.info(
                "Episode %d: subtask %s%s frames [%d, %d] -> sampled %s",
                actual_episode_idx,
                subtask_index,
                label,
                start,
                end,
                frames,
            )
        return selected_frames

    def _subtask_stride_frames(
        self, frame_metadata: dict[int, dict[str, Any]]
    ) -> int:
        stride_frames = self.sampling_config.get("stride_frames")
        stride_seconds = self.sampling_config.get("stride_seconds")
        if stride_frames is not None and stride_seconds is not None:
            raise ValueError("Set only one of sampling.stride_frames or stride_seconds")
        if stride_frames is not None:
            stride = int(stride_frames)
        elif stride_seconds is not None:
            fps = self.dataset_fps or self._infer_fps(frame_metadata)
            if fps is None:
                raise ValueError(
                    "sampling.stride_seconds requires dataset fps or timestamps"
                )
            stride = int(round(float(stride_seconds) * fps))
        else:
            raise ValueError(
                "sampling.mode=subtask_aware requires sampling.stride_frames "
                "or sampling.stride_seconds"
            )
        if stride < 1:
            raise ValueError("Sampling stride must be at least 1 frame")
        return stride

    def _infer_fps(self, frame_metadata: dict[int, dict[str, Any]]) -> float | None:
        timestamps = [
            metadata.get("timestamp")
            for _, metadata in sorted(frame_metadata.items())
            if metadata.get("timestamp") is not None
        ]
        if len(timestamps) < 2:
            return None
        deltas = np.diff(np.array(timestamps, dtype=np.float64))
        positive = deltas[deltas > 0]
        if len(positive) == 0:
            return None
        return float(1.0 / np.median(positive))

    def _frame_metadata_for_episode(
        self, row: Any, length: int
    ) -> dict[int, dict[str, Any]]:
        actual_episode_idx = int(row["episode_index"])
        metadata = {
            frame_idx: {
                "frame_idx": frame_idx,
                "subtask_index": None,
                "timestamp": None,
            }
            for frame_idx in range(length)
        }

        data_path = self._episode_data_path(row)
        if data_path is None or not data_path.exists():
            logger.warning("Episode %d: data parquet not found", actual_episode_idx)
            return metadata

        columns = ["episode_index", "frame_index", "timestamp", "subtask_index"]
        schema_names = set(pq.read_schema(data_path).names)
        read_columns = [column for column in columns if column in schema_names]
        if "frame_index" not in read_columns:
            logger.warning("Episode %d: data parquet has no frame_index", actual_episode_idx)
            return metadata

        if data_path not in self._data_metadata_cache:
            self._data_metadata_cache[data_path] = pq.read_table(
                data_path, columns=read_columns
            ).to_pandas()

        frame_df = self._data_metadata_cache[data_path]
        if "episode_index" in frame_df.columns:
            frame_df = frame_df[frame_df["episode_index"] == actual_episode_idx]
        frame_df = frame_df.sort_values("frame_index")

        for _, frame_row in frame_df.iterrows():
            frame_idx = int(frame_row["frame_index"])
            if frame_idx < 0 or frame_idx >= length:
                continue
            timestamp = None
            if "timestamp" in frame_row.index and not pd.isna(frame_row["timestamp"]):
                timestamp = float(frame_row["timestamp"])
            subtask_index = None
            if (
                "subtask_index" in frame_row.index
                and not pd.isna(frame_row["subtask_index"])
            ):
                subtask_index = int(frame_row["subtask_index"])
            metadata[frame_idx] = {
                "frame_idx": frame_idx,
                "subtask_index": subtask_index,
                "timestamp": timestamp,
            }
            if subtask_index is not None and subtask_index in self.subtask_info:
                metadata[frame_idx].update(self.subtask_info[subtask_index])

        return metadata

    def _episode_data_path(self, row: Any) -> Path | None:
        chunk_key = "data/chunk_index"
        file_key = "data/file_index"
        if chunk_key not in row.index or file_key not in row.index:
            return None
        return (
            self.dataset_path
            / "data"
            / f"chunk-{int(row[chunk_key]):03d}"
            / f"file-{int(row[file_key]):03d}.parquet"
        )

    def _load_dataset_fps(self) -> float | None:
        info_path = self.dataset_path / "meta" / "info.json"
        if not info_path.exists():
            return None
        try:
            with open(info_path) as f:
                info = json.load(f)
            fps = info.get("fps")
            return float(fps) if fps else None
        except Exception as e:
            logger.warning("Could not read dataset fps from %s: %s", info_path, e)
            return None

    def _load_subtask_info(self) -> dict[int, dict[str, Any]]:
        subtask_path = self.dataset_path / "meta" / "subtask.parquet"
        if not subtask_path.exists():
            return {}
        try:
            df = pq.read_table(subtask_path).to_pandas()
        except Exception as e:
            logger.warning("Could not read subtask metadata from %s: %s", subtask_path, e)
            return {}
        info: dict[int, dict[str, Any]] = {}
        for _, row in df.iterrows():
            subtask_index = int(row["subtask_index"])
            info[subtask_index] = {
                "atomic_skill": str(row.get("atomic_skill", "")),
                "subtask": str(row.get("subtask", "")),
                "has_regrasp": bool(row.get("has_regrasp", False)),
            }
        return info

    def _episode_instruction(self, row: Any) -> str:
        source = self.instruction_config.get("instruction_source", "episode_field")
        if source == "none":
            default = self._fallback_instruction()
            logger.warning(
                "Instruction source is disabled; falling back to %r",
                default,
            )
            return default
        if source == "join_file":
            return self._joined_instruction(row)

        field = self.instruction_config.get("instruction_field", "expand_task")
        if field in row.index:
            text = self._coerce_instruction(row[field])
            if text:
                return text

        default = self._fallback_instruction()
        logger.warning(
            "Episode %s has no instruction field %r; falling back to %r",
            row.get("episode_index", "<unknown>"),
            field,
            default,
        )
        return default

    @staticmethod
    def _coerce_instruction(value: Any) -> str:
        """Coerce an instruction cell to text.

        Handles plain string fields (e.g. AnyGrasp `expand_task`) as well as
        the list-valued `tasks` column used by standard LeRobot v3 datasets,
        where an episode may carry one or more natural-language task strings.
        Joining the parts lets a single episode with multiple tasks still yield
        a usable instruction for the discovery layer.
        """
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (list, tuple, np.ndarray)):
            parts = [str(item).strip() for item in value if str(item).strip()]
            return " ".join(parts)
        return ""

    def _joined_instruction(self, row: Any) -> str:
        if self.join_instructions is None:
            return self._fallback_instruction()

        episode_key = self.instruction_config.get(
            "instruction_join_episode_key", "episode_index"
        )
        key = row[episode_key]
        text = self.join_instructions.get(key)
        if text:
            return text

        default = self._fallback_instruction()
        logger.warning("No joined instruction for %s=%r; falling back", episode_key, key)
        return default

    def _load_join_instructions(self) -> dict[Any, str]:
        join_path = self.instruction_config.get("instruction_join_path")
        if join_path is None:
            raise ValueError("instruction_source=join_file requires instruction_join_path")
        path = Path(join_path)
        if not path.is_absolute():
            path = self.dataset_path / path

        file_key = self.instruction_config.get("instruction_join_file_key", "episode_index")
        text_field = self.instruction_config.get(
            "instruction_join_text_field", "expand_task"
        )
        table = pq.read_table(path, columns=[file_key, text_field]).to_pandas()
        return {
            row[file_key]: str(row[text_field]).strip()
            for _, row in table.iterrows()
            if str(row[text_field]).strip()
        }

    def _fallback_instruction(self) -> str:
        return str(self.instruction_config.get("default_instruction", "") or "")

    def _load_camera_frames(
        self, row: Any, camera_name: str, frame_indices: list[int]
    ) -> dict[int, np.ndarray]:
        chunk_index = int(row[f"videos/{camera_name}/chunk_index"])
        file_index = int(row[f"videos/{camera_name}/file_index"])
        start_index = int(row["dataset_from_index"])
        video_path = (
            self.dataset_path
            / "videos"
            / camera_name
            / f"chunk-{chunk_index:03d}"
            / f"file-{file_index:03d}.mp4"
        )
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        frames: dict[int, np.ndarray] = {}
        for frame_idx in frame_indices:
            frames[frame_idx] = self._extract_frame(video_path, start_index + frame_idx)
        return frames

    def _extract_frame(self, video_path: Path, global_frame_idx: int) -> np.ndarray:
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(video_path),
                    "-vf",
                    f"select=eq(n\\,{global_frame_idx})",
                    "-vframes",
                    "1",
                    tmp.name,
                ],
                check=True,
            )
            return np.array(Image.open(tmp.name).convert("RGB"))
