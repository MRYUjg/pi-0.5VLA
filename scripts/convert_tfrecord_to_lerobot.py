#!/usr/bin/env python3
"""Convert libero_spatial RLDS TFRecord data to LeRobot v2.0 format.

Reads TFRecords directly (bypasses TFDS, works with incomplete shards).

Usage:
    # 默认输出到 ~/.cache/huggingface/lerobot/（训练脚本自动读取）
    python scripts/convert_tfrecord_to_lerobot.py --data_dir libreo_raw_data --repo_id local/libero_spatial

    # 输出到指定目录
    python scripts/convert_tfrecord_to_lerobot.py --data_dir libreo_raw_data --repo_id local/libero_spatial --output_dir /tmp/my_data

    # dry-run 检查数据格式
    python scripts/convert_tfrecord_to_lerobot.py --data_dir libreo_raw_data --dry_run
"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import tqdm
import tyro


def main(
    data_dir: str,
    repo_id: str = "local/libero_spatial",
    output_dir: str | None = None,
    *,
    dry_run: bool = False,
    push_to_hub: bool = False,
):
    """Convert libero_spatial TFRecords to LeRobot format.

    Args:
        data_dir: Path to the directory containing libero_spatial_no_noops/
        repo_id: LeRobot dataset name (e.g. "your_org/my_dataset")
        output_dir: Output directory for LeRobot data. Defaults to <project_root>/data/
        dry_run: If True, only read and print the first record's structure
        push_to_hub: If True, push to HuggingFace Hub after conversion
    """
    # --- Imports done here to fail early with clear messages ---
    try:
        import tensorflow as tf
    except ImportError:
        print("tensorflow not installed. Run: pip install tensorflow-cpu")
        sys.exit(1)

    try:
        import cv2
    except ImportError:
        print("opencv-python not installed. Run: pip install opencv-python")
        sys.exit(1)

    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    # --- Locate TFRecord files ---
    data_dir = Path(data_dir)
    tfrecord_dir = data_dir / "libero_spatial_no_noops" / "1.0.0"
    if not tfrecord_dir.exists():
        alt_dir = data_dir / "1.0.0"
        if alt_dir.exists():
            tfrecord_dir = alt_dir
        else:
            print(f"TFRecord directory not found. Tried:")
            print(f"  {data_dir / 'libero_spatial_no_noops' / '1.0.0'}")
            print(f"  {data_dir / '1.0.0'}")
            sys.exit(1)

    tfrecord_files = sorted(tfrecord_dir.glob("*.tfrecord-*"))
    if not tfrecord_files:
        print(f"No .tfrecord files found in {tfrecord_dir}")
        sys.exit(1)

    print(f"Found {len(tfrecord_files)} TFRecord file(s)")
    for f in tfrecord_files:
        print(f"  {f.name}  ({f.stat().st_size / 1024 / 1024:.1f} MB)")

    # --- Dry run: inspect first record ---
    if dry_run:
        _inspect_record(tfrecord_files[0])
        return

    # --- Determine output path ---
    if output_dir is not None:
        # User-specified output directory: override LeRobot home
        import lerobot.common.datasets.lerobot_dataset as _lds
        _lds.HF_LEROBOT_HOME = Path(output_dir).resolve()
        output_root = Path(output_dir)
    else:
        # Default: use LeRobot's global default (~/.cache/huggingface/lerobot/)
        output_root = HF_LEROBOT_HOME

    output_path = output_root / repo_id
    print(f"Output: {output_path}")

    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=10,
        features={
            "image":       {"dtype": "image",   "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image",   "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "state":       {"dtype": "float32", "shape": (8,),          "names": ["state"]},
            "actions":     {"dtype": "float32", "shape": (7,),          "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # --- Convert ---
    total_episodes = 0
    total_frames = 0
    for tf_file in tfrecord_files:
        raw_dataset = tf.data.TFRecordDataset(str(tf_file), compression_type="")
        for raw_record in tqdm.tqdm(raw_dataset, desc=f"Converting {tf_file.name}", unit="episodes"):
            frames, task = _parse_episode(raw_record.numpy())
            if frames is None:
                continue
            for frame in frames:
                frame["task"] = task
                dataset.add_frame(frame)
            dataset.save_episode()
            total_episodes += 1
            total_frames += len(frames)

    # Consolidate (method name varies by LeRobot version)
    if hasattr(dataset, "consolidate"):
        dataset.consolidate()
    elif hasattr(dataset, "finalize"):
        dataset.finalize()
    else:
        print("Note: LeRobot version does not support consolidate/finalize, skipping")
    print(f"\nDone: {total_episodes} episodes, {total_frames} frames -> {output_path}")

    if push_to_hub:
        dataset.push_to_hub(tags=["libero", "panda", "rlds"], private=False, push_videos=True, license="apache-2.0")
        print("Pushed to HuggingFace Hub")


def _parse_episode(raw_bytes: bytes):
    """Parse a single tf.train.Example from raw TFRecord bytes.

    The LIBERO RLDS dataset stores episodes as Example protos where each
    step field is a variable-length list packed into a single feature.

    Returns (list_of_frame_dicts, task_string) or (None, None) on failure.
    """
    import tensorflow as tf
    import cv2

    features = {
        "steps/action":                    tf.io.VarLenFeature(tf.float32),
        "steps/observation/image":         tf.io.VarLenFeature(tf.string),
        "steps/observation/wrist_image":   tf.io.VarLenFeature(tf.string),
        "steps/observation/state":         tf.io.VarLenFeature(tf.float32),
        "steps/language_instruction":      tf.io.VarLenFeature(tf.string),
    }

    try:
        parsed = tf.io.parse_single_example(raw_bytes, features)
    except Exception as e:
        print(f"  Skipping record (parse error): {e}")
        return None, None

    # Action: flat float list, 7 values per step
    action_flat = tf.sparse.to_dense(parsed["steps/action"]).numpy()
    if len(action_flat) == 0:
        return None, None
    num_steps = len(action_flat) // 7
    if num_steps == 0:
        return None, None
    actions = action_flat.reshape(num_steps, 7)

    # State: flat float list, 8 values per step
    state_flat = tf.sparse.to_dense(parsed["steps/observation/state"]).numpy()
    state = state_flat.reshape(num_steps, 8)

    # Language instruction: take first non-empty string
    lang_sparse = tf.sparse.to_dense(parsed["steps/language_instruction"], default_value=b"").numpy()
    task = ""
    for lb in lang_sparse:
        s = lb.decode("utf-8") if isinstance(lb, bytes) else str(lb)
        if s:
            task = s
            break

    # Images: list of JPEG bytes, one per step
    img_bytes_list = tf.sparse.to_dense(parsed["steps/observation/image"]).numpy()
    wrist_bytes_list = tf.sparse.to_dense(parsed["steps/observation/wrist_image"]).numpy()

    # Verify lengths match
    if len(img_bytes_list) != num_steps or len(wrist_bytes_list) != num_steps:
        print(f"  Warning: image count mismatch (imgs={len(img_bytes_list)},"
              f" wrists={len(wrist_bytes_list)}, steps={num_steps}), trimming")
        num_steps = min(num_steps, len(img_bytes_list), len(wrist_bytes_list))
        actions = actions[:num_steps]
        state = state[:num_steps]

    frames = []
    for i in range(num_steps):
        base_img = cv2.cvtColor(
            cv2.imdecode(np.frombuffer(img_bytes_list[i], np.uint8), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )
        wrist_img = cv2.cvtColor(
            cv2.imdecode(np.frombuffer(wrist_bytes_list[i], np.uint8), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )

        frames.append({
            "image":       base_img,
            "wrist_image": wrist_img,
            "state":       state[i].astype(np.float32),
            "actions":     actions[i].astype(np.float32),
        })

    return frames, task


def _inspect_record(tfrecord_file: Path):
    """Print the schema of the first record in a TFRecord file."""
    import tensorflow as tf

    raw_dataset = tf.data.TFRecordDataset(str(tfrecord_file), compression_type="")
    for raw_record in raw_dataset.take(1):
        raw_bytes = raw_record.numpy()

        # Try as Example
        try:
            example = tf.train.Example()
            example.ParseFromString(raw_bytes)
            print("=== tf.train.Example ===")
            for k, v in example.features.feature.items():
                kind = v.WhichOneof("kind")
                print(f"  {k}: {kind}")
        except Exception:
            pass

        # Try as SequenceExample
        try:
            seq_example = tf.train.SequenceExample()
            seq_example.ParseFromString(raw_bytes)
            print("\n=== tf.train.SequenceExample ===")
            print("Context features:")
            for k, v in seq_example.context.feature.items():
                print(f"  {k}: {v.WhichOneof('kind')}")
            print("Sequence features:")
            for k, v in seq_example.feature_lists.feature_list.items():
                print(f"  {k}: len={len(v.feature)}")
        except Exception as e:
            print(f"SequenceExample parse failed: {e}")

        # Try RLDS parse
        print("\n=== Attempting RLDS parse ===")
        frames, task = _parse_episode(raw_bytes)
        if frames is not None:
            print(f"RLDS parse OK: {len(frames)} frames, task='{task}'")
            f0 = frames[0]
            print(f"  image shape:       {f0['image'].shape} {f0['image'].dtype}")
            print(f"  wrist_image shape: {f0['wrist_image'].shape} {f0['wrist_image'].dtype}")
            print(f"  state shape:       {f0['state'].shape} {f0['state'].dtype}")
            print(f"  actions shape:     {f0['actions'].shape} {f0['actions'].dtype}")
            print(f"\n  state[:8]  = {f0['state']}")
            print(f"  actions[:7] = {f0['actions']}")


if __name__ == "__main__":
    tyro.cli(main)