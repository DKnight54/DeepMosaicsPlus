"""
Auto-bucket real censored/uncensored video pairs by detected mosaic size.

Given two folders — one of censored clips, one of matching uncensored clips,
each pair sharing the SAME filename across the two folders — this:
  1. Extracts frames from both videos into origin_image/ (uncensored) and
     mosaic_image/ (censored) — the exact layout util/dataloader.py's
     real-pair training path expects (see util.dataloader.VideoLoader).
  2. Runs the same block-size detector used by tools/test_mosaic_size.py on
     the censored frames to estimate the mosaic's size as % of frame short
     side.
  3. Copies the resulting pair folder into <output_dir>/bucket_<N>/<name>/,
     where N is the nearest bucket in your --model_manifest.

IMPORTANT — alignment: this assumes each pair's two videos are already the
same content, frame-for-frame, just with/without censoring (same cuts, same
timing). It does NOT verify that for you. Misaligned pairs are worse than
no real-pair data at all (see the earlier discussion) — spot check a few
extracted frame pairs before trusting a bucket's real-pair data in training.

Folder layout — see datasets/raw/README.md for the full picture:
    datasets/raw/censored/clip1.mp4
    datasets/raw/uncensored/clip1.mp4      <- same filename, different folder
    datasets/raw/censored/scene04.mp4
    datasets/raw/uncensored/scene04.mp4

Usage:
    python tools/bucket_real_pairs.py \
        --censored_dir datasets/raw/censored \
        --uncensored_dir datasets/raw/uncensored \
        --output_dir datasets/real_pairs \
        --mosaic_position_model_path mosaic_position.pth \
        --model_manifest pretrained_models/mosaic/size_manifest.json
"""
import os
import sys


# Windows consoles often default to a non-UTF-8 code page (cp1252/cp437),
# which mangles Unicode characters like em-dashes into '?' or '�'. Force
# UTF-8 stdout so any character in this script's output prints correctly
# regardless of the system's console code page.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import shutil
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import cv2
from tqdm import tqdm
from cores import Options
from models import loadmodel, runmodel
from util import mosaic as mosaic_util
from util import ffmpeg as ffmpeg_util
from util import util as util_
from util import image_processing as impro
from models.yolo_mosaic_detector import YoloMosaicDetector


def find_pairs(censored_dir, uncensored_dir):
    censored_files = {f for f in os.listdir(censored_dir) if util_.is_video(os.path.join(censored_dir, f))}
    uncensored_files = {f for f in os.listdir(uncensored_dir) if util_.is_video(os.path.join(uncensored_dir, f))}
    common = sorted(censored_files & uncensored_files)
    pairs = []
    for f in common:
        name, _ext = os.path.splitext(f)
        pairs.append((name, os.path.join(censored_dir, f), os.path.join(uncensored_dir, f)))

    only_censored = censored_files - uncensored_files
    only_uncensored = uncensored_files - censored_files
    if only_censored:
        print(f"Note: {len(only_censored)} file(s) in censored/ have no matching filename in "
              f"uncensored/, skipped: {sorted(only_censored)}")
    if only_uncensored:
        print(f"Note: {len(only_uncensored)} file(s) in uncensored/ have no matching filename in "
              f"censored/, skipped: {sorted(only_uncensored)}")
    return pairs


def _edge_median(values, window):
    """Sliding median with edge replication (util.filt.medfilt pads with zeros,
    which drags the first/last positions toward 0)."""
    if window <= 1 or len(values) < window:
        return values
    pad = window // 2
    padded = np.pad(values, pad, mode='edge')
    return np.array([np.median(padded[i:i + window]) for i in range(len(values))], dtype=values.dtype)


def crop_pair_to_mosaic(netM, opt, origin_dir, mosaic_dir, outsize, ex_range=(1.2, 1.7)):
    """Crop every frame of a pair to the mosaic region and resize to outsize x outsize.

    Extraction leaves FULL frames. The synthetic dataset (make_video_dataset.py)
    and the cleaner at inference both work on a crop around the mosaic region,
    so full frames teach the network the wrong thing: the mosaic is a small part
    of the training crop and its blocks are the wrong size relative to what it
    sees later (4x off for small mosaics), and a random crop of a wide frame
    often misses the mosaic altogether.

    The box is located on the CENSORED frames (that's where the mosaic is), and
    the SAME box is applied to the matching uncensored frame, so the pair stays
    pixel-aligned. Positions are median-filtered over time to remove jitter,
    gaps carry the nearest detection, and the box is kept square and fully
    inside the frame. Returns stats, or None if no mosaic was found.
    """
    names = sorted(f for f in os.listdir(mosaic_dir) if f.lower().endswith('.jpg'))
    n_orig = len([f for f in os.listdir(origin_dir) if f.lower().endswith('.jpg')])
    names = names[:min(len(names), n_orig)]
    n = len(names)
    if n == 0:
        return None
    min_size = getattr(opt, 'min_mosaic_size', 100)
    pos = np.zeros((n, 3), dtype=np.int64)            # x, y, half-width of the detected box
    
    use_yolo_batch = isinstance(netM, YoloMosaicDetector) and hasattr(netM, 'detect_batch')
    batch_size = 16

    pbar_loc = tqdm(total=n, desc='  locating mosaic', leave=False)
    for start_i in range(0, n, batch_size):
        chunk_names = names[start_i:start_i + batch_size]
        chunk_imgs = []
        valid_indices = []

        for offset, name in enumerate(chunk_names):
            frame = impro.imread(os.path.join(mosaic_dir, name))
            if frame is not None:
                chunk_imgs.append(frame)
                valid_indices.append(start_i + offset)

        if chunk_imgs:
            if use_yolo_batch:
                dets = netM.detect_batch(chunk_imgs, opt)
            else:
                dets = [runmodel.get_mosaic_position(im, netM, opt) for im in chunk_imgs]

            for idx, (mask, x, y, size) in zip(valid_indices, dets):
                if mask is not None and size > min_size:
                    pos[idx] = (x, y, size)

        pbar_loc.update(len(chunk_names))
    pbar_loc.close()
    valid = pos[:, 2] > 0
    if not valid.any():
        return None
    idx = np.arange(n)
    prev_i = np.maximum.accumulate(np.where(valid, idx, -1))
    next_i = np.minimum.accumulate(np.where(valid, idx, n)[::-1])[::-1]
    for i in np.where(~valid)[0]:
        cands = [j for j in (prev_i[i], next_i[i]) if 0 <= j < n]
        pos[i] = pos[min(cands, key=lambda j: abs(j - i))]
    win = getattr(opt, 'medfilt_num', 5)
    win = win if win % 2 == 1 else win + 1
    for c in range(3):
        pos[:, c] = _edge_median(pos[:, c], win)

    ex_mul = random.uniform(*ex_range)               # same margin range as the synthetic builder
    sides, size_mismatch = [], False
    for i, name in enumerate(tqdm(names, desc='  Cropping Mosaic area', leave=False)):
        mosaic = impro.imread(os.path.join(mosaic_dir, name))
        origin = impro.imread(os.path.join(origin_dir, name))
        h, w = mosaic.shape[:2]
        if origin.shape[:2] != (h, w):               # pair at different resolutions: align to the censored one
            origin = cv2.resize(origin, (w, h), interpolation=cv2.INTER_AREA)
            size_mismatch = True
        half = max(8, min(int(pos[i, 2] * ex_mul), h // 2, w // 2))
        cx = int(np.clip(pos[i, 0], half, w - half))
        cy = int(np.clip(pos[i, 1], half, h - half))
        box = (slice(cy - half, cy + half), slice(cx - half, cx + half))
        interp = cv2.INTER_AREA if 2 * half > outsize else cv2.INTER_CUBIC
        impro.imwrite(os.path.join(mosaic_dir, name), cv2.resize(mosaic[box], (outsize, outsize), interpolation=interp))
        impro.imwrite(os.path.join(origin_dir, name), cv2.resize(origin[box], (outsize, outsize), interpolation=interp))
        sides.append(2 * half)
    # drop frames that couldn't be paired: leaving them would put uncropped full frames in the dataset
    for d in (origin_dir, mosaic_dir):
        for f in tqdm(os.listdir(d), desc=f'Disposing mismatch images'):
            if f.lower().endswith('.jpg') and f not in set(names):
                os.remove(os.path.join(d, f))
    return {'frames': n, 'median_side': int(np.median(sides)), 'ex_mul': ex_mul,
            'enlarged_pct': 100.0 * float(np.mean(np.array(sides) < outsize)), 'resized_origin': size_mismatch}


def extract_frames(videopath, out_dir, fps, start_sec=None, duration_sec=None):
    """Extract frames from `videopath` into `out_dir` (%05d.jpg).
    If `start_sec` and `duration_sec` are provided, extracts only that scene window.
    """
    util_.makedirs(out_dir)
    out_pattern = os.path.join(out_dir, '%05d.jpg')
    if start_sec is not None and duration_sec is not None:
        start_stamp = util_.second2stamp(int(start_sec))
        last_stamp = util_.second2stamp(max(1, int(round(duration_sec))))
        ffmpeg_util.video2image(
            videopath,
            out_pattern,
            fps=fps,
            start_time=start_stamp,
            last_time=last_stamp,
        )
    else:
        ffmpeg_util.video2image(videopath, out_pattern, fps=fps)


def detect_pct(opt, netM, censored_video, n_samples, min_pct, max_pct,
               scan_stride_sec=2.0, max_gap_sec=5.0, min_scene_sec=3.0):
    """Single-pass sequential scan using cap.grab().
    Discovers continuous censored scenes and estimates mosaic block % per scene.
    Returns a list of scene dicts:
      [{'start': float, 'duration': float, 'pct': float,
        'confidence': float, 'n_nonsquare': int, 'n_samples': int}, ...]
    """
    cap = cv2.VideoCapture(censored_video)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    step = max(1, int(round(fps * scan_stride_sec)))
    min_size = getattr(opt, 'min_mosaic_size', 100)
    min_conf = getattr(opt, 'auto_model_min_confidence', 0.15)
    max_samples_per_scene = max(n_samples, 12)
    use_yolo = isinstance(netM, YoloMosaicDetector)

    hits = []  # list of (timestamp_sec, frame, mask)
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc='  scanning video', unit='frame', leave=False)
    while True:
        if not cap.grab():
            break

        if frame_idx % step == 0:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                orig_h, orig_w = frame.shape[:2]
                max_side = max(orig_h, orig_w)

                if use_yolo and max_side > 640:
                    scale = 640.0 / max_side
                    scan_frame = cv2.resize(
                        frame,
                        (int(round(orig_w * scale)), int(round(orig_h * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                    mask_small, x, y, size = netM.detect(scan_frame, opt)
                    size = int(round(size / scale))
                    if mask_small is not None and size > min_size:
                        mask = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    else:
                        mask = None
                elif use_yolo:
                    mask, x, y, size = netM.detect(frame, opt)
                else:
                    mask, x, y, size = runmodel.get_mosaic_position(frame, netM, opt)

                if mask is not None and size > min_size:
                    hits.append((frame_idx / fps, frame, mask))
                    pbar.set_postfix(hits=len(hits), refresh=False)

        frame_idx += 1
        pbar.update(1)

    pbar.close()

    cap.release()
    if not hits:
        return []

    # Group adjacent hits into continuous scenes
    raw_scenes = []
    cur = [hits[0]]
    for item in hits[1:]:
        if item[0] - cur[-1][0] <= max_gap_sec:
            cur.append(item)
        else:
            raw_scenes.append(cur)
            cur = [item]
    raw_scenes.append(cur)

    # Estimate mosaic block % individually for each scene
    scenes = []
    video_dur = total_frames / fps
    for group in raw_scenes:
        start_sec = max(0.0, group[0][0] - (scan_stride_sec * 0.5))
        end_sec = min(video_dur, group[-1][0] + (scan_stride_sec * 0.5))
        duration_sec = end_sec - start_sec

        if duration_sec < min_scene_sec:
            continue

        if len(group) > max_samples_per_scene:
            idxs = np.linspace(0, len(group) - 1, max_samples_per_scene, dtype=int)
            sampled = [group[i] for i in idxs]
        else:
            sampled = group

        pairs = [(f, m) for (_ts, f, m) in sampled]
        res = mosaic_util.estimate_mosaic_block_pct_multi(
            pairs,
            min_confidence=min_conf,
            min_pct=min_pct,
            max_pct=max_pct,
        )
        if res is None:
            continue

        scenes.append({
            'start': start_sec,
            'duration': duration_sec,
            'pct': res['pct'],
            'confidence': res['confidence'],
            'n_nonsquare': res.get('n_nonsquare', 0),
            'n_samples': len(pairs),
        })

    return scenes


def maybe_reuse_uncensored(pairs, mode, synthetic_source_dir):
    """Optionally copy each pair's uncensored clip into a shared synthetic-
    source folder, so it can ALSO be used as raw material for synthetic
    mosaic generation at other bucket sizes (make_video_dataset.py + train's
    --dataset_mosaic_pct) — not just the one real mosaic size this specific
    pair happens to have. Worthwhile if your real pairs for a bucket come
    from a small number of distinct sources: this multiplies effective
    content diversity per bucket without needing brand-new source footage.
    Skips files already copied (safe to re-run).
    """
    if mode == 'no':
        return
    if mode == 'ask':
        print(f"\nYou have {len(pairs)} uncensored clip(s). These can ALSO be reused as source "
              f"footage for synthetic mosaic generation at any bucket size (not just whatever "
              f"real size these particular pairs have) — useful if a bucket's real pairs come "
              f"from only a few distinct videos, since it adds content diversity for free.")
        answer = input(f"Copy them into {synthetic_source_dir} for that purpose? [y/N]: ").strip().lower()
        if answer not in ('y', 'yes'):
            print("Skipping — real pairs will only be used for their own detected bucket.")
            return
    os.makedirs(synthetic_source_dir, exist_ok=True)
    copied = 0
    for _name, _censored_path, uncensored_path in pairs:
        dst = os.path.join(synthetic_source_dir, os.path.basename(uncensored_path))
        if os.path.exists(dst):
            continue
        shutil.copy2(uncensored_path, dst)
        copied += 1
    print(f"Copied {copied} new uncensored clip(s) into {synthetic_source_dir}.")
    print(f"Note: make_datasets/make_video_dataset.py reprocesses everything in its --datadir on "
          f"every run and does not skip already-done videos — if you've already built a synthetic "
          f"pool from an earlier version of this folder, either point --startcnt past your existing "
          f"output count, or move already-processed source clips elsewhere before re-running it, "
          f"to avoid overwriting previous output.")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--censored_dir', required=True, help='folder of real censored clips')
    parser.add_argument('--uncensored_dir', required=True, help='folder of matching uncensored clips (same filenames)')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--fps', type=int, default=15,
        help='extraction frame rate for both videos in each pair. Match this to whatever --fps '
             'you use for make_datasets/make_video_dataset.py when building your synthetic pool, '
             'if combining both into the same bucket for training (train/clean/train.py\'s '
             'temporal model uses a frame-count stride, so mixed extraction rates across your '
             'combined dataset mean inconsistent real-world motion per training sample).')
    parser.add_argument('--n_samples', type=int, default=8)
    parser.add_argument('--outsize', type=int, default=572,
        help='frames are cropped to the mosaic region and resized to this size (px). Use the same value as '
             'make_datasets/make_video_dataset.py --outsize; 572 for the default 512px training, 286 for 256px.')
    parser.add_argument('--no_crop', action='store_true',
        help='keep full frames instead of cropping to the mosaic region (not recommended, see crop_pair_to_mosaic)')
    parser.add_argument('--reuse_uncensored_for_synthetic', choices=['ask', 'yes', 'no'], default='ask',
        help='Also copy each pair\'s uncensored clip into --synthetic_source_dir, so it can be '
             'reused as raw material for synthetic mosaic generation at OTHER bucket sizes (via '
             'make_datasets/make_video_dataset.py + --dataset_mosaic_pct) — not just the one real '
             'size this pair happens to have. Worth doing if your real pairs for a bucket come '
             'from few distinct sources, since it multiplies content diversity per bucket without '
             'new source footage. \'ask\' (default) prompts once interactively; \'yes\'/\'no\' skip '
             'the prompt, useful for scripting.')
             
    parser.add_argument('--scan_stride_sec', type=float, default=2.0,
                        help='seconds between sampled frames during scene discovery')
    parser.add_argument('--max_gap_sec', type=float, default=5.0,
                        help='max uncensored gap (seconds) before splitting into a new scene')
    parser.add_argument('--min_scene_sec', type=float, default=3.0,
                        help='minimum censored scene duration (seconds) to keep')
    parser.add_argument('--yolo_conf', type=float, default=0.35,
                        help='confidence threshold for YOLO mosaic detector (.pt/.onnx)')
    parser.add_argument('--synthetic_source_dir', default=os.path.join('datasets', 'raw', 'synthetic_source'))
    known, remaining = parser.parse_known_args()

    # Rebuild sys.argv with just the args Options() understands, plus the
    # --model_path shim (same reasoning as tools/test_mosaic_size.py: this
    # tool never runs the clean/generator model, only position detection).
    sys.argv = [sys.argv[0]] + remaining
    if '--model_path' not in sys.argv and '--mosaic_position_model_path' in sys.argv:
        pos_idx = sys.argv.index('--mosaic_position_model_path')
        sys.argv += ['--model_path', sys.argv[pos_idx + 1], '--mode', 'clean', '--netG', 'unet_128']

    opt = Options().getparse(test_flag=True)
    opt.yolo_conf = known.yolo_conf

    if not os.path.isdir(known.censored_dir):
        print(f"--censored_dir not found: {known.censored_dir}")
        sys.exit(1)
    if not os.path.isdir(known.uncensored_dir):
        print(f"--uncensored_dir not found: {known.uncensored_dir}")
        sys.exit(1)

    print("Loading mosaic-position model...")
    if opt.mosaic_position_model_path.lower().endswith(('.pt', '.onnx')):
        netM = YoloMosaicDetector(opt.mosaic_position_model_path, conf_thresh=0.35)
    else:
        netM = loadmodel.bisenet(opt, 'mosaic')

    manifest_path = opt.model_manifest
    if not os.path.exists(manifest_path):
        print(f"--model_manifest not found at {manifest_path}. Create it first (see "
              f"pretrained_models/mosaic/README.md) so this tool knows your bucket boundaries.")
        sys.exit(1)
    manifest = loadmodel.load_size_manifest(manifest_path)
    min_pct, max_pct = min(manifest.keys()), max(manifest.keys())
    print(f"Bucket range from manifest: {min_pct:g}%-{max_pct:g}%")

    pairs = find_pairs(known.censored_dir, known.uncensored_dir)
    if not pairs:
        print("No matching filenames found between --censored_dir and --uncensored_dir.")
        sys.exit(1)
    print(f"Found {len(pairs)} pair(s).")

    maybe_reuse_uncensored(pairs, known.reuse_uncensored_for_synthetic, known.synthetic_source_dir)

    results = []
    for name, censored_path, uncensored_path in pairs:
        print(f"\n--- {name} ---")
        scenes = detect_pct(
            opt, netM, censored_path, known.n_samples, min_pct, max_pct,
            scan_stride_sec=known.scan_stride_sec,
            max_gap_sec=known.max_gap_sec,
            min_scene_sec=known.min_scene_sec,
        )

        if not scenes:
            print("  Could not confidently detect any censored scenes, skipping.")
            continue
            
            
        print(f"  Found {len(scenes)} censored scene(s).")
        for scene_idx, sc in enumerate(scenes):
            _, bucket = loadmodel.select_model_by_pct(sc['pct'], manifest_path)
            clip_name = f"{name}_scene{scene_idx:03d}"

            note = ""
            if sc ['n_nonsquare'] > 0:
                note = f"  ({sc ['n_nonsquare']}/{sc ['n_samples']} samples non-square)"
            print(f"  detected pct={sc ['pct']:.2f}%  confidence={sc ['confidence']:.2f}{note}  -> bucket {bucket:g}%")

            bucket_dir = os.path.join(known.output_dir, f'bucket_{bucket:g}', clip_name)
            origin_dir = os.path.join(bucket_dir, 'origin_image')
            mosaic_dir = os.path.join(bucket_dir, 'mosaic_image')
            print(f"  extracting frames -> {bucket_dir}")
            extract_frames(uncensored_path, origin_dir, known.fps, start_sec=sc['start'], duration_sec=sc['duration'])
            extract_frames(censored_path, mosaic_dir, known.fps, start_sec=sc['start'], duration_sec=sc['duration'])

            n_orig = len(os.listdir(origin_dir))
            n_mosaic = len(os.listdir(mosaic_dir))
            if n_orig != n_mosaic:
                print(f"  ** WARNING: frame count mismatch (origin={n_orig}, mosaic={n_mosaic}) — "
                      f"the two source videos may differ in length/framerate. This pair's alignment "
                      f"is suspect; recommend checking manually before training on it. **")

            if not known.no_crop:
                stats = crop_pair_to_mosaic(netM, opt, origin_dir, mosaic_dir, known.outsize)
                if stats is None:
                    print("  could not locate the mosaic in this pair's frames -- skipping it")
                    shutil.rmtree(bucket_dir, ignore_errors=True)
                    continue
                print(f"  cropped {stats['frames']} frame pairs to the mosaic region: median source crop "
                      f"{stats['median_side']}px -> {known.outsize}px ({stats['enlarged_pct']:.0f}% enlarged)")
                if stats['resized_origin']:
                    print("  note: the uncensored video has a different resolution than the censored one; "
                          "it was resized to match before cropping.")

            results.append((clip_name, sc ['pct'], bucket))

    print("\n=== Summary ===")
    for name, pct, bucket in results:
        print(f"  {name}: {pct:.2f}% -> bucket {bucket:g}%")
    print(f"\nDone. {len(results)} scene(s) from {len(pairs)} pair(s) bucketed under {known.output_dir}/bucket_<N>/")
    print("Recommended next step: spot-check a few frame pairs in each bucket folder for alignment "
          "before training (see this script's docstring).")


if __name__ == '__main__':
    main()
