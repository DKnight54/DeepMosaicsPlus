import os
import shutil
import cv2
import numpy as np

MIN_FRAMES = 100          # Matches train.py --M 100 minimum clip length
BOTTOM_BAND_RATIO = 0.65  # Check bottom 35% of the 512x512 crop where subtitles enter


def has_subtitle_in_crop(orig_bgr, mos_bgr):
    """Detect high-contrast burned-in text present in origin_image but absent in mosaic_image."""
    h, w = orig_bgr.shape[:2]
    y0 = int(h * BOTTOM_BAND_RATIO)

    orig_gray = cv2.cvtColor(orig_bgr[y0:, :], cv2.COLOR_BGR2GRAY).astype(np.float32)
    mos_gray = cv2.cvtColor(mos_bgr[y0:, :], cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Burned-in subtitles are bright (> 215) in origin while much darker in mosaic (> 70 diff)
    bright_diff = (orig_gray > 215.0) & ((orig_gray - mos_gray) > 70.0)
    bright_ratio = float(np.mean(bright_diff))

    # Also check sharp horizontal text stroke edges present only in origin_image
    orig_gx = np.abs(cv2.Sobel(orig_gray, cv2.CV_32F, 1, 0, ksize=3))
    mos_gx = np.abs(cv2.Sobel(mos_gray, cv2.CV_32F, 1, 0, ksize=3))
    text_edges = bright_diff & ((orig_gx - mos_gx) > 80.0)
    edge_ratio = float(np.mean(text_edges))

    return bright_ratio > 0.004 and edge_ratio > 0.0015


def split_clean_runs(flags, min_len):
    """Return list of (start_idx, end_idx_exclusive) contiguous False runs >= min_len."""
    runs = []
    start = None
    for i, bad in enumerate(flags):
        if not bad and start is None:
            start = i
        elif bad and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    if start is not None and len(flags) - start >= min_len:
        runs.append((start, len(flags)))
    return runs


def process_real_pairs_folder(target_dir, prefix_filter="", dry_run=False):
    """Scan bucket_* subfolders under target_dir and split/clean subtitled scenes."""
    total_checked = 0
    total_kept_clips = 0
    total_removed_clips = 0
    total_frames_scanned = 0
    total_frames_kept = 0

    for bucket in sorted(os.listdir(target_dir)):
        bdir = os.path.join(target_dir, bucket)
        if not os.path.isdir(bdir) or not bucket.startswith("bucket_"):
            continue

        for clip in sorted(os.listdir(bdir)):
            if "_dup" in clip:
                continue
            if prefix_filter and not clip.startswith(prefix_filter):
                continue

            cdir = os.path.join(bdir, clip)
            orig_dir = os.path.join(cdir, "origin_image")
            mos_dir = os.path.join(cdir, "mosaic_image")
            if not (os.path.isdir(orig_dir) and os.path.isdir(mos_dir)):
                continue

            names = sorted(f for f in os.listdir(orig_dir) if f.lower().endswith(".jpg"))
            n_total = len(names)
            if n_total < MIN_FRAMES:
                continue

            total_checked += 1
            total_frames_scanned += n_total
            sub_flags = []
            for fname in names:
                o_img = cv2.imread(os.path.join(orig_dir, fname))
                m_img = cv2.imread(os.path.join(mos_dir, fname))
                if o_img is None or m_img is None:
                    sub_flags.append(True)
                else:
                    sub_flags.append(has_subtitle_in_crop(o_img, m_img))

            n_bad = sum(sub_flags)
            bad_pct = 100.0 * n_bad / n_total

            if n_bad == 0:
                print(f"[clean] {bucket}/{clip}: {n_total} frames — 0.0% subtitled (100.0% kept)")
                total_kept_clips += 1
                total_frames_kept += n_total
                continue

            runs = split_clean_runs(sub_flags, MIN_FRAMES)
            kept_in_scene = sum(e_i - s_i for s_i, e_i in runs)
            kept_pct = 100.0 * kept_in_scene / n_total
            total_frames_kept += kept_in_scene

            print(
                f"[split] {bucket}/{clip}: {n_bad}/{n_total} ({bad_pct:.1f}%) subtitled -> "
                f"{len(runs)} sub-clip(s) >= {MIN_FRAMES}f ({kept_in_scene}/{n_total} frames, {kept_pct:.1f}% kept)"
            )

            if dry_run:
                total_kept_clips += len(runs)
                if not runs:
                    total_removed_clips += 1
                continue

            created_dirs = []
            for r_idx, (s_i, e_i) in enumerate(runs):
                new_name = f"{clip}_sub{r_idx}"
                new_dir = os.path.join(bdir, new_name)
                os.makedirs(os.path.join(new_dir, "origin_image"), exist_ok=True)
                os.makedirs(os.path.join(new_dir, "mosaic_image"), exist_ok=True)

                for out_idx, src_idx in enumerate(range(s_i, e_i), start=1):
                    src_fname = names[src_idx]
                    dst_fname = f"{out_idx:05d}.jpg"
                    shutil.copy2(os.path.join(orig_dir, src_fname), os.path.join(new_dir, "origin_image", dst_fname))
                    shutil.copy2(os.path.join(mos_dir, src_fname), os.path.join(new_dir, "mosaic_image", dst_fname))
                created_dirs.append(new_dir)

            shutil.rmtree(cdir, ignore_errors=True)
            total_kept_clips += len(created_dirs)
            if not created_dirs:
                total_removed_clips += 1

    overall_frame_pct = (100.0 * total_frames_kept / total_frames_scanned) if total_frames_scanned else 0.0
    print(
        f"\nSummary: checked {total_checked} scene(s) ({total_frames_scanned:,} frames) -> "
        f"{total_kept_clips} clean clip(s) kept ({total_frames_kept:,} frames, {overall_frame_pct:.1f}% survival), "
        f"{total_removed_clips} scene(s) fully discarded."
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split real-pair scenes around burned-in subtitles.")
    parser.add_argument("--real_pairs_dir", default=os.path.join("datasets", "real_pair"), help="Path to real_pair folder")
    parser.add_argument("--prefix", default="", help="Optional video prefix filter (e.g. 'SubbedVideoName')")
    parser.add_argument("--dry_run", action="store_true", help="Preview splits without modifying folders")
    args = parser.parse_args()

    process_real_pairs_folder(args.real_pairs_dir, prefix_filter=args.prefix, dry_run=args.dry_run)
