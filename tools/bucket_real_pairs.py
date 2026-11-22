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
import glob
import time
import shutil
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

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
from tools.filter_subtitled_scenes import has_subtitle_in_pair, split_clean_runs, MIN_FRAMES as SUB_MIN_FRAMES


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
    """Sliding median with edge replication."""
    if window <= 1 or len(values) < window:
        return values
    pad = window // 2
    padded = np.pad(values, pad, mode='edge')
    return np.array([np.median(padded[i:i + window]) for i in range(len(values))], dtype=values.dtype)


def crop_pair_to_mosaic(netM, opt, origin_dir, mosaic_dir, outsize,
                        ex_range=(1.2, 1.7), min_frames=100,
                        gpu_lock=None, show_pbar=True):
    """Crop every frame of a pair to the mosaic region and resize to outsize x outsize."""
    names = sorted(f for f in os.listdir(mosaic_dir) if f.lower().endswith('.jpg'))
    n_orig = len([f for f in os.listdir(origin_dir) if f.lower().endswith('.jpg')])
    names = names[:min(len(names), n_orig)]
    n = len(names)
    if n < min_frames or n == 0:
        return None

    min_size = getattr(opt, 'min_mosaic_size', 100)
    pos = np.zeros((n, 3), dtype=np.int64)

    use_yolo_batch = isinstance(netM, YoloMosaicDetector) and hasattr(netM, 'detect_batch')
    batch_size = 16

    pbar_loc = tqdm(total=n, desc='  locating mosaic', leave=False, disable=not show_pbar)
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
            if gpu_lock is not None:
                with gpu_lock:
                    if use_yolo_batch:
                        dets = netM.detect_batch(chunk_imgs, opt)
                    else:
                        dets = [runmodel.get_mosaic_position(im, netM, opt) for im in chunk_imgs]
            else:
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

    ex_mul = random.uniform(*ex_range)
    sides, size_mismatch = [], False
    name_set = set(names)

    for i, name in enumerate(tqdm(names, desc='  Cropping Mosaic area', leave=False, disable=not show_pbar)):
        mosaic = impro.imread(os.path.join(mosaic_dir, name))
        origin = impro.imread(os.path.join(origin_dir, name))
        h, w = mosaic.shape[:2]
        if origin.shape[:2] != (h, w):
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

    for d in (origin_dir, mosaic_dir):
        for f in os.listdir(d):
            if f.lower().endswith('.jpg') and f not in name_set:
                os.remove(os.path.join(d, f))

    return {
        'frames': n,
        'median_side': int(np.median(sides)),
        'ex_mul': ex_mul,
        'enlarged_pct': 100.0 * float(np.mean(np.array(sides) < outsize)),
        'resized_origin': size_mismatch,
    }

def _split_subtitles_in_place(bucket_dir, min_frames=SUB_MIN_FRAMES):
    """Split a newly cropped pair folder into _sub0, _sub1... if burned-in subtitles are detected."""
    orig_dir = os.path.join(bucket_dir, 'origin_image')
    mos_dir = os.path.join(bucket_dir, 'mosaic_image')
    names = sorted(f for f in os.listdir(orig_dir) if f.lower().endswith('.jpg'))
    n = len(names)
    if n < min_frames:
        return 0

    raw = []
    for fn in names:
        oi = cv2.imread(os.path.join(orig_dir, fn))
        mi = cv2.imread(os.path.join(mos_dir, fn))
        raw.append(True if (oi is None or mi is None) else has_subtitle_in_pair(oi, mi))

    flags = [False] * n
    i = 0
    while i < n:
        if raw[i]:
            j = i
            while j < n and raw[j]:
                j += 1
            if (j - i) >= 5:
                for k in range(max(0, i - 2), min(n, j + 2)):
                    flags[k] = True
            i = j
        else:
            i += 1

    if sum(flags) == 0:
        return 1

    runs = split_clean_runs(flags, min_frames)
    for r_idx, (s_i, e_i) in enumerate(runs):
        sub_dir = f"{bucket_dir}_sub{r_idx}"
        os.makedirs(os.path.join(sub_dir, 'origin_image'), exist_ok=True)
        os.makedirs(os.path.join(sub_dir, 'mosaic_image'), exist_ok=True)
        for out_i, src_i in enumerate(range(s_i, e_i), start=1):
            shutil.copy2(os.path.join(orig_dir, names[src_i]), os.path.join(sub_dir, 'origin_image', f"{out_i:05d}.jpg"))
            shutil.copy2(os.path.join(mos_dir, names[src_i]), os.path.join(sub_dir, 'mosaic_image', f"{out_i:05d}.jpg"))
    shutil.rmtree(bucket_dir, ignore_errors=True)
    return len(runs)

def extract_frames(videopath, out_dir, fps, start_sec=None, duration_sec=None):
    """Extract frames from `videopath` into `out_dir` (%05d.jpg)."""
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
               scan_stride_sec=2.0, max_gap_sec=5.0, min_scene_sec=3.0,
               gpu_lock=None, show_pbar=True, progress_cb=None):
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

    pbar = tqdm(total=total_frames, desc='  scanning video', unit='frame', leave=False, disable=not show_pbar)
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
                    if gpu_lock is not None:
                        with gpu_lock:
                            mask_small, x, y, size = netM.detect(scan_frame, opt)
                    else:
                        mask_small, x, y, size = netM.detect(scan_frame, opt)
                    size = int(round(size / scale))
                    if mask_small is not None and size > min_size:
                        mask = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    else:
                        mask = None
                elif use_yolo:
                    if gpu_lock is not None:
                        with gpu_lock:
                            mask, x, y, size = netM.detect(frame, opt)
                    else:
                        mask, x, y, size = netM.detect(frame, opt)
                else:
                    if gpu_lock is not None:
                        with gpu_lock:
                            mask, x, y, size = runmodel.get_mosaic_position(frame, netM, opt)
                    else:
                        mask, x, y, size = runmodel.get_mosaic_position(frame, netM, opt)

                if mask is not None and size > min_size:
                    hits.append((frame_idx / fps, frame, mask))
                    if show_pbar:
                        pbar.set_postfix(hits=len(hits), refresh=False)

        frame_idx += 1
        pbar.update(1)
        if progress_cb is not None and (frame_idx % (step * 5) == 0):
            progress_cb(int(100 * frame_idx / max(1, total_frames)))

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
    if mode == 'no':
        return
    if mode == 'ask':
        answer = input(f"Copy uncensored clips into {synthetic_source_dir} for synthetic generation? [y/N]: ").strip().lower()
        if answer not in ('y', 'yes'):
            return
    os.makedirs(synthetic_source_dir, exist_ok=True)
    copied = 0
    for _name, _censored_path, uncensored_path in pairs:
        dst = os.path.join(synthetic_source_dir, os.path.basename(uncensored_path))
        if os.path.exists(dst):
            continue
        shutil.copy2(uncensored_path, dst)
        copied += 1
    if copied:
        print(f"Copied {copied} new uncensored clip(s) into {synthetic_source_dir}.")


def _cleanup_partial_pair(output_dir, pair_name):
    """Remove any leftover scene folders for a pair that did not finish cleanly."""
    pattern = os.path.join(output_dir, 'bucket_*', f"{pair_name}_scene*")
    for d in glob.glob(pattern):
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)


def _find_existing_pair_scenes(output_dir, pair_name):
    """Find already-extracted scene folders on disk for pairs built before .done markers existed."""
    found = []
    pattern = os.path.join(output_dir, 'bucket_*', f"{pair_name}_scene*")
    for d in sorted(glob.glob(pattern)):
        orig_d = os.path.join(d, 'origin_image')
        mos_d = os.path.join(d, 'mosaic_image')
        if os.path.isdir(orig_d) and os.path.isdir(mos_d):
            n_o = len([f for f in os.listdir(orig_d) if f.lower().endswith('.jpg')])
            n_m = len([f for f in os.listdir(mos_d) if f.lower().endswith('.jpg')])
            if n_o >= 100 and n_o == n_m:
                bucket_folder = os.path.basename(os.path.dirname(d))
                try:
                    b_val = float(bucket_folder.replace('bucket_', ''))
                except ValueError:
                    b_val = 0.0
                found.append((os.path.basename(d), 0.0, b_val))
    return found


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--censored_dir', required=True, help='folder of real censored clips')
    parser.add_argument('--uncensored_dir', required=True, help='folder of matching uncensored clips (same filenames)')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--min_frames', type=int, default=100)
    parser.add_argument('--n_samples', type=int, default=8)
    parser.add_argument('--outsize', type=int, default=572)
    parser.add_argument('--no_crop', action='store_true')
    parser.add_argument('--reuse_uncensored_for_synthetic', choices=['ask', 'yes', 'no'], default='ask')
    parser.add_argument('--scan_stride_sec', type=float, default=2.0)
    parser.add_argument('--max_gap_sec', type=float, default=5.0)
    parser.add_argument('--min_scene_sec', type=float, default=3.0)
    parser.add_argument('--yolo_conf', type=float, default=0.35)
    parser.add_argument('--synthetic_source_dir', default=os.path.join('datasets', 'raw', 'synthetic_source'))
    parser.add_argument('--workers', type=int, default=2,
                        help='parallel worker threads for FFmpeg extraction & cropping (default: 2)')
    parser.add_argument('--no_resume', action='store_true',
                        help='reprocess all pairs from scratch instead of skipping already-completed pairs')
    parser.add_argument('--gui_progress', action='store_true',
                        help='emit machine-readable @@PROGRESS lines and suppress tqdm bars for GUI log cleanliness')
    known, remaining = parser.parse_known_args()

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
        netM = YoloMosaicDetector(opt.mosaic_position_model_path, conf_thresh=known.yolo_conf)
    else:
        netM = loadmodel.bisenet(opt, 'mosaic')

    manifest_path = opt.model_manifest
    if not os.path.exists(manifest_path):
        print(f"--model_manifest not found at {manifest_path}.")
        sys.exit(1)
    manifest = loadmodel.load_size_manifest(manifest_path)
    min_pct, max_pct = min(manifest.keys()), max(manifest.keys())
    print(f"Bucket range from manifest: {min_pct:g}%-{max_pct:g}%")

    pairs = find_pairs(known.censored_dir, known.uncensored_dir)
    if not pairs:
        print("No matching filenames found between --censored_dir and --uncensored_dir.")
        sys.exit(1)

    maybe_reuse_uncensored(pairs, known.reuse_uncensored_for_synthetic, known.synthetic_source_dir)

    os.makedirs(known.output_dir, exist_ok=True)
    done_dir = os.path.join(known.output_dir, '.done')
    os.makedirs(done_dir, exist_ok=True)

    all_pair_names = [p[0] for p in pairs]
    if known.no_resume:
        for name in all_pair_names:
            sentinel = os.path.join(done_dir, f"{name}.done")
            if os.path.exists(sentinel):
                os.remove(sentinel)
            _cleanup_partial_pair(known.output_dir, name)
        todo_pairs = pairs
        skipped_count = 0
    else:
        todo_pairs = []
        skipped_count = 0
        existing_by_pair = {}
        latest_mtime = -1.0
        latest_pair = None
        for name, _, _ in pairs:
            sentinel = os.path.join(done_dir, f"{name}.done")
            if os.path.exists(sentinel):
                continue
            scenes_on_disk = _find_existing_pair_scenes(known.output_dir, name)
            if scenes_on_disk:
                existing_by_pair[name] = scenes_on_disk
                for sc_name, _, b_val in scenes_on_disk:
                    p_dir = os.path.join(known.output_dir, f"bucket_{b_val:g}", sc_name)
                    mt = os.path.getmtime(p_dir)
                    if mt > latest_mtime:
                        latest_mtime = mt
                        latest_pair = name

        if existing_by_pair:
            for name, scenes_on_disk in existing_by_pair.items():
                if name == latest_pair:
                    print(f"Resume check: re-verifying last in-flight pair '{name}'...")
                    _cleanup_partial_pair(known.output_dir, name)
                else:
                    with open(os.path.join(done_dir, f"{name}.done"), 'w', encoding='utf-8') as f:
                        f.write(f"adopted {len(scenes_on_disk)} scene(s)\n")

        for item in pairs:
            name = item[0]
            if os.path.exists(os.path.join(done_dir, f"{name}.done")):
                skipped_count += 1
            else:
                _cleanup_partial_pair(known.output_dir, name)
                todo_pairs.append(item)

    workers = max(1, min(int(known.workers), 8))
    use_gui = bool(known.gui_progress)
    show_inner_pbar = (workers == 1) and (not use_gui)

    print(f"Found {len(pairs)} pair(s) total — {skipped_count} already completed, "
          f"{len(todo_pairs)} to process (workers={workers}).")

    if not todo_pairs:
        print("All pairs are already completed!")
        return

    gpu_lock = threading.Lock()
    state_lock = threading.Lock()
    slot_pool = list(range(workers))
    worker_slots = {}  # slot_idx -> "W1: name [stage]@pct"
    completed_pairs = 0
    total_todo = len(todo_pairs)
    results = []

    def emit_gui_progress(msg=""):
        if not use_gui:
            return
        with state_lock:
            w_str = ";".join(worker_slots[k] for k in sorted(worker_slots))
            done_n = completed_pairs
        print(f"@@PROGRESS|done={done_n}|total={total_todo}|workers={w_str}|msg={msg}", flush=True)

    def set_worker_status(slot, name, stage, pct=None):
        short = name if len(name) <= 22 else name[:19] + "..."
        pct_part = f"@{pct}" if pct is not None else "@"
        with state_lock:
            worker_slots[slot] = f"W{slot + 1}: {short} [{stage}]{pct_part}"
        emit_gui_progress()

    def process_one_pair(pair_tuple):
        nonlocal completed_pairs
        name, censored_path, uncensored_path = pair_tuple
        t0 = time.time()

        with state_lock:
            slot = slot_pool.pop(0)
        set_worker_status(slot, name, "scan", 0)

        pair_logs = []
        pair_results = []
        try:
            scenes = detect_pct(
                opt, netM, censored_path, known.n_samples, min_pct, max_pct,
                scan_stride_sec=known.scan_stride_sec,
                max_gap_sec=known.max_gap_sec,
                min_scene_sec=known.min_scene_sec,
                gpu_lock=gpu_lock,
                show_pbar=show_inner_pbar,
                progress_cb=lambda p: set_worker_status(slot, name, "scan", p),
            )

            if not scenes:
                pair_logs.append(f"  {name}: no censored scenes detected, skipped.")
            else:
                kept = 0
                for scene_idx, sc in enumerate(scenes):
                    est_frames = int(round(sc['duration'] * known.fps))
                    if est_frames < known.min_frames:
                        continue

                    set_worker_status(slot, name, f"scene {scene_idx + 1}/{len(scenes)}")
                    _, bucket = loadmodel.select_model_by_pct(sc['pct'], manifest_path)
                    clip_name = f"{name}_scene{scene_idx:03d}"
                    bucket_dir = os.path.join(known.output_dir, f'bucket_{bucket:g}', clip_name)
                    origin_dir = os.path.join(bucket_dir, 'origin_image')
                    mosaic_dir = os.path.join(bucket_dir, 'mosaic_image')

                    extract_frames(uncensored_path, origin_dir, known.fps,
                                   start_sec=sc['start'], duration_sec=sc['duration'])
                    extract_frames(censored_path, mosaic_dir, known.fps,
                                   start_sec=sc['start'], duration_sec=sc['duration'])

                    n_orig = len(os.listdir(origin_dir))
                    n_mosaic = len(os.listdir(mosaic_dir))
                    warn = ""
                    if n_orig != n_mosaic:
                        warn = f" [WARN: frame mismatch orig={n_orig} vs mos={n_mosaic}]"

                    if not known.no_crop:
                        stats = crop_pair_to_mosaic(
                            netM, opt, origin_dir, mosaic_dir, known.outsize,
                            min_frames=known.min_frames,
                            gpu_lock=gpu_lock,
                            show_pbar=show_inner_pbar,
                        )
                        if stats is None:
                            shutil.rmtree(bucket_dir, ignore_errors=True)
                            continue
                        n_clean_clips = _split_subtitles_in_place(bucket_dir, min_frames=known.min_frames)
                        if n_clean_clips == 0:
                            continue
                        sub_note = f" -> {n_clean_clips} clean sub-clip(s)" if n_clean_clips > 1 else ""
                        pair_logs.append(
                            f"    scene {scene_idx:03d}: pct={sc['pct']:.2f}% (conf={sc['confidence']:.2f}) "
                            f"-> bucket {bucket:g}% | {stats['frames']} frames, median {stats['median_side']}px{warn}{sub_note}"
                        )
                    else:
                        pair_logs.append(
                            f"    scene {scene_idx:03d}: pct={sc['pct']:.2f}% (conf={sc['confidence']:.2f}) "
                            f"-> bucket {bucket:g}% | {n_orig} full frames{warn}"
                        )

                    pair_results.append((clip_name, sc['pct'], bucket))
                    kept += 1

                if kept == 0:
                    pair_logs.append(f"  {name}: {len(scenes)} scene(s) found, none met min_frames={known.min_frames}.")

            with open(os.path.join(done_dir, f"{name}.done"), 'w', encoding='utf-8') as f:
                f.write(f"ok scenes={len(pair_results)}\n")

        finally:
            dt = time.time() - t0
            with state_lock:
                worker_slots.pop(slot, None)
                slot_pool.append(slot)
                slot_pool.sort()
                completed_pairs += 1
                idx_now = completed_pairs
                results.extend(pair_results)

            header = f"[{idx_now}/{total_todo}] {name} ({dt:.1f}s) -> {len(pair_results)} clip(s)"
            print("\n".join([header] + pair_logs), flush=True)
            emit_gui_progress(f"last: {name} ({len(pair_results)} clip(s))")

    emit_gui_progress("starting workers")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_one_pair, p) for p in todo_pairs]
        for fut in as_completed(futures):
            fut.result()

    print("\n=== Summary ===")
    bucket_counts = {}
    for clip_name, pct, b in results:
        print(f"  {clip_name}: {pct:.2f}% -> bucket {b:g}%")
        bucket_counts[b] = bucket_counts.get(b, 0) + 1
    for b in sorted(bucket_counts):
        print(f"  Bucket {b:g}%: +{bucket_counts[b]} scene(s)")
    print(f"\nDone. {len(results)} new scene(s) from {total_todo} pair(s) bucketed under {known.output_dir}/bucket_<N>/")


if __name__ == '__main__':
    main()