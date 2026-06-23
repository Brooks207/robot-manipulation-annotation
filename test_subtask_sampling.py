"""Test script to verify subtask-aware frame sampling."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from annotation.lerobot_v3_dataset import LeRobotV3Dataset

# Test configuration
dataset_path = Path("./data/svla_so100_sorting")
camera_names = ["observation.images.top"]

# Uniform sampling for comparison
print("=" * 60)
print("TEST 1: Uniform sampling (baseline)")
print("=" * 60)
dataset_uniform = LeRobotV3Dataset(
    dataset_path=dataset_path,
    camera_names=camera_names,
    instruction_config={"instruction_source": "episode_field", "instruction_field": "expand_task"},
    episode_indices=[0],
    sampling_config={"mode": "uniform", "frames_per_episode": 10},
    load_frames=False,
)

episode = dataset_uniform.get_episode(0)
print(f"Episode {episode['episode_index']}: {episode['num_frames']} total frames")
print(f"Selected frames ({len(episode['frame_indices'])}): {episode['frame_indices']}")
print()

# Subtask-aware sampling
print("=" * 60)
print("TEST 2: Subtask-aware sampling")
print("=" * 60)
dataset_subtask = LeRobotV3Dataset(
    dataset_path=dataset_path,
    camera_names=camera_names,
    instruction_config={"instruction_source": "episode_field", "instruction_field": "expand_task"},
    episode_indices=[0],
    sampling_config={
        "mode": "subtask_aware",
        "stride_frames": 30,
        "keep_subtask_boundaries": True,
    },
    load_frames=False,
)

episode = dataset_subtask.get_episode(0)
print(f"Episode {episode['episode_index']}: {episode['num_frames']} total frames")
print(f"Selected frames ({len(episode['frame_indices'])}): {episode['frame_indices'][:20]}...")
print()

# Print per-subtask breakdown
frame_metadata = episode["frame_metadata"]
subtask_groups = {}
for frame_idx in episode["frame_indices"]:
    subtask_idx = frame_metadata[frame_idx].get("subtask_index", -1)
    if subtask_idx not in subtask_groups:
        subtask_groups[subtask_idx] = []
    subtask_groups[subtask_idx].append(frame_idx)

print(f"Frames grouped by subtask:")
for subtask_idx in sorted(subtask_groups.keys()):
    frames = subtask_groups[subtask_idx]
    print(f"  Subtask {subtask_idx}: {len(frames)} frames - {frames[:10]}...")

print()
print("Test completed successfully!")
