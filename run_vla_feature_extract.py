#!/usr/bin/env python3
"""Extract frozen SigLIP-SO400M (OpenVLA vision encoder) features for LIBERO mask instances.

For each annotated mask instance (episode_idx, frame_idx, instance_id):
  - Crop the instance bounding box from the RGB frame (+ 12 px padding)
  - Resize to 384×384 and pass through the frozen SigLIP-SO400M vision encoder
  - Take the pooler output (mean-pooled patch tokens) → 1152-dim feature vector

Indices are aligned exactly with probe_features.npz so that
run_representation_probe_vla.py can join the two files without re-sorting.

Must run server-side (needs the original LeRobot v3 video frames).
Output vla_features.npz can be brought back locally for the probe comparison.

Usage:
    python run_vla_feature_extract.py \\
        --dataset-path ./data/libero \\
        --masks outputs/libero_smoke/segmentation/masks.parquet \\
        --camera-name observation.images.image \\
        --probe-cache outputs/representation_probe/probe_features.npz \\
        --out outputs/vla_features.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import torch
from PIL import Image
from transformers import SiglipModel, SiglipProcessor

sys.path.insert(0, str(Path(__file__).parent))
from annotation.lerobot_v3_dataset import LeRobotV3Dataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SIGLIP_MODEL_ID = "google/siglip-so400m-patch14-384"
HIDDEN_DIM = 1152  # SO400M hidden size


def decode_mask(row: pd.Series) -> np.ndarray:
    rle = {
        "counts": row["rle_counts"].encode("utf-8")
        if isinstance(row["rle_counts"], str)
        else row["rle_counts"],
        "size": list(row["rle_size"]),
    }
    return mask_util.decode(rle).astype(bool)


def crop_instance(frame_rgb: np.ndarray, mask: np.ndarray, pad: int = 12) -> Image.Image | None:
    """Tight bounding-box crop of the masked region, padded and returned as PIL."""
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    r1 = max(0, int(rows.min()) - pad)
    r2 = min(frame_rgb.shape[0] - 1, int(rows.max()) + pad)
    c1 = max(0, int(cols.min()) - pad)
    c2 = min(frame_rgb.shape[1] - 1, int(cols.max()) + pad)
    crop = frame_rgb[r1 : r2 + 1, c1 : c2 + 1]
    return Image.fromarray(crop)


def extract_siglip_features(
    model: SiglipModel,
    processor: SiglipProcessor,
    crops: list[Image.Image],
    device: torch.device,
) -> np.ndarray:
    """Return pooler_output (mean-pooled patch tokens) for a batch of crops. (B, 1152)."""
    inputs = processor(images=crops, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_out = model.vision_model(pixel_values=inputs["pixel_values"].to(model.dtype))
    return vision_out.pooler_output.float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--camera-name", required=True)
    parser.add_argument("--probe-cache", type=Path, required=True,
                         help="probe_features.npz from run_representation_probe.py; "
                              "used to determine which instances to extract and align indices.")
    parser.add_argument("--out", type=Path, default=Path("outputs/vla_features.npz"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    # Load probe cache to get the canonical ordered list of instances.
    cache = np.load(args.probe_cache, allow_pickle=True)
    episode_idxs = cache["episode_idx"]
    frame_idxs = cache["frame_idx"]
    instance_ids = cache["instance_id"]
    y = cache["y"]
    n = len(y)
    logger.info("Probe cache: %d instances across %d categories", n, len(set(y)))

    # Build index: (episode, frame, instance_id) → position in the arrays above.
    key_to_pos: dict[tuple[int, int, int], int] = {
        (int(episode_idxs[i]), int(frame_idxs[i]), int(instance_ids[i])): i for i in range(n)
    }

    masks_df = pd.read_parquet(args.masks)

    # Which frames do we need per episode?
    needed: dict[int, set[int]] = defaultdict(set)
    for ep, fr in zip(episode_idxs, frame_idxs):
        needed[int(ep)].add(int(fr))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading SigLIP-SO400M on %s...", device)
    processor = SiglipProcessor.from_pretrained(SIGLIP_MODEL_ID)
    model = SiglipModel.from_pretrained(SIGLIP_MODEL_ID, torch_dtype=torch.float16).to(device)
    model.eval()
    logger.info("SigLIP ready.")

    siglip_X = np.full((n, HIDDEN_DIM), np.nan, dtype=np.float32)
    total_extracted = 0

    for ep_idx, frames_needed in sorted(needed.items()):
        logger.info("Episode %d: %d frames", ep_idx, len(frames_needed))
        try:
            ds = LeRobotV3Dataset(
                dataset_path=args.dataset_path,
                camera_names=[args.camera_name],
                instruction_config={"instruction_source": "none"},
                episode_indices=[ep_idx],
                frame_indices=sorted(frames_needed),
                load_frames=True,
            )
            episode = ds.get_episode(0)
            frame_map: dict[int, np.ndarray | None] = episode["frames"][args.camera_name]
        except Exception as exc:
            logger.warning("Episode %d failed to load (%s); skipping.", ep_idx, exc)
            continue

        for fr_idx in sorted(frames_needed):
            frame_rgb = frame_map.get(fr_idx)
            if frame_rgb is None:
                logger.warning("Episode %d frame %d: no RGB; skipping.", ep_idx, fr_idx)
                continue

            sel = (masks_df["episode_idx"] == ep_idx) & (masks_df["frame_idx"] == fr_idx)
            frame_masks = masks_df[sel]
            if frame_masks.empty:
                continue

            # Collect crops and their positions in one pass; batch through SigLIP.
            crops: list[Image.Image] = []
            positions: list[int] = []
            for _, row in frame_masks.iterrows():
                key = (ep_idx, fr_idx, int(row["instance_id"]))
                if key not in key_to_pos:
                    continue
                mask = decode_mask(row)
                crop = crop_instance(frame_rgb, mask)
                if crop is None:
                    continue
                crops.append(crop)
                positions.append(key_to_pos[key])

            for batch_start in range(0, len(crops), args.batch_size):
                batch_crops = crops[batch_start : batch_start + args.batch_size]
                batch_pos = positions[batch_start : batch_start + args.batch_size]
                feats = extract_siglip_features(model, processor, batch_crops, device)
                for pos, feat in zip(batch_pos, feats):
                    siglip_X[pos] = feat
                total_extracted += len(batch_crops)

    n_missing = int(np.isnan(siglip_X).any(axis=1).sum())
    logger.info("Extracted %d/%d features (%d missing).", total_extracted, n, n_missing)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        siglip_X=siglip_X,
        y=y,
        episode_idx=episode_idxs,
        frame_idx=frame_idxs,
        instance_id=instance_ids,
    )
    logger.info("Saved → %s", args.out)


if __name__ == "__main__":
    main()
