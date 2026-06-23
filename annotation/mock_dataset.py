"""Mock LeRobot dataset for testing pipeline before real data integration."""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class MockLeRobotDataset:
    """
    Mock dataset that simulates LeRobot v3.0 format.

    Used for testing pipeline logic before actual LeRobot integration.
    """

    def __init__(self, dataset_path: Path, num_episodes: int = 3, frames_per_episode: int = 10):
        """
        Initialize mock dataset.

        Args:
            dataset_path: Path to dataset (not used, for API compatibility)
            num_episodes: Number of mock episodes to generate
            frames_per_episode: Number of frames per episode
        """
        self.dataset_path = dataset_path
        self.num_episodes = num_episodes
        self.frames_per_episode = frames_per_episode

        logger.warning(
            f"Using MockLeRobotDataset with {num_episodes} episodes, "
            f"{frames_per_episode} frames each"
        )

    def __len__(self) -> int:
        """Return number of episodes."""
        return self.num_episodes

    def get_episode(self, episode_idx: int) -> dict:
        """
        Get episode data.

        Args:
            episode_idx: Episode index

        Returns:
            Dictionary with episode data:
                - instruction: Task instruction string
                - num_frames: Number of frames
                - frames: Dict[camera_name, List[np.ndarray]] - RGB frames
        """
        # Mock instructions
        instructions = [
            "pick up the red cup and place it on the table",
            "grasp the blue block and stack it on the green block",
            "open the drawer and retrieve the spoon",
        ]

        instruction = instructions[episode_idx % len(instructions)]

        # Generate mock RGB frames (H=480, W=640, C=3)
        frames_dict = {
            "observation.images.top": [
                self._generate_mock_frame(episode_idx, frame_idx)
                for frame_idx in range(self.frames_per_episode)
            ]
        }

        return {
            "instruction": instruction,
            "num_frames": self.frames_per_episode,
            "frames": frames_dict,
        }

    def _generate_mock_frame(self, episode_idx: int, frame_idx: int) -> np.ndarray:
        """
        Generate a mock RGB frame with deterministic noise.

        Args:
            episode_idx: Episode index (for seeding)
            frame_idx: Frame index (for seeding)

        Returns:
            RGB frame of shape (480, 640, 3), dtype uint8
        """
        # Deterministic seed for reproducibility
        rng = np.random.RandomState(seed=episode_idx * 1000 + frame_idx)

        # Generate colored noise (different color per episode)
        base_color = (episode_idx * 80) % 256
        frame = rng.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
        frame[:, :, episode_idx % 3] = (frame[:, :, episode_idx % 3] * 0.5 + base_color).astype(np.uint8)

        return frame
