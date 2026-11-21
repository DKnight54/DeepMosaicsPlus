import os

import json
import shutil
import argparse
import subprocess
import cv2
import numpy as np
from tqdm import tqdm


VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".ts", ".m4v", ".webm")


def extract_robust_fingerprint(gray_hw, w=128, h=72):
    """Extract normalized full-action fingerprint (including center silhouette) while excluding
    top corner watermarks and bottom subtitle strip. Downsampling to 32x18 naturally averages
    out mosaic blocks while catching pose changes and jump cuts in the center of the frame."""
    # Crop out top 12% (corner watermarks) and bottom 26% (burned-in subtitles)
    roi = gray_hw[int(h * 0.12):int(h * 0.74), int(w * 0.04):int(w * 0.96)]
    coarse_center = cv2.resize(roi, (48, 27), interpolation=cv2.INTER_AREA)

    # Also keep sharp edge gradients on outer-mid zones for fine alignment
    y1, y2 = int(h * 0.16), int(h * 0.72)
    left_strip = gray_hw[y1:y2, int(w * 0.04):int(w * 0.22)]
    right_strip = gray_hw[y1:y2, int(w * 0.78):int(w * 0.96)]

    feat = np.concatenate([coarse_center.ravel() * 1.5, left_strip.ravel(), right_strip.ravel()])
    std = float(np.std(feat))
    if std > 1e-3:
        return (feat - float(np.mean(feat))) / std
    return np.zeros_like(feat)


def index_video_ffmpeg(video_path, cache_dir=None, target_fps=6.0, w=128, h=72):
    """Fast FFmpeg C-pipeline fingerprint indexing with optional disk .npz cache."""
    st = os.stat(video_path)
    cache_file = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        safe_name = f"{os.path.basename(video_path)}_{st.st_size}_{int(target_fps)}fps.npz"
        cache_file = os.path.join(cache_dir, safe_name)
        if os.path.isfile(cache_file):
            data = np.load(cache_file)
            print(f"  loaded cached index for {os.path.basename(video_path)} ({len(data['mat'])} frames)")
            return data["mat"], data["ts"], float(data["src_fps"])

    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    est_samples = max(1, int(round((total_frames / src_fps) * target_fps)))
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", video_path,
        "-vf", f"fps={target_fps},scale={w}:{h}",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = w * h
    feats = []
    timestamps = []

    pbar = tqdm(total=est_samples, desc=f"  indexing {os.path.basename(video_path)}", unit="f")
    idx = 0
    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        gray = np.frombuffer(raw, dtype=np.uint8).reshape((h, w)).astype(np.float32)
        feats.append(extract_robust_fingerprint(gray, w=w, h=h))
        timestamps.append(idx / target_fps)
        idx += 1
        pbar.update(1)

    pbar.close()
    proc.stdout.close()
    proc.wait()

    mat = np.ascontiguousarray(np.vstack(feats), dtype=np.float32) if feats else np.empty((0, 1), dtype=np.float32)
    ts_arr = np.array(timestamps, dtype=np.float64)
    if cache_file and len(mat) > 0:
        np.savez(cache_file, mat=mat, ts=ts_arr, src_fps=np.array(src_fps))
    return mat, ts_arr, src_fps


def fmt_tc(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60.0
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def save_qc_strip(cen_path, unc_path, cen_start, unc_start, dur, out_png):
    """Save 3-row side-by-side contact sheet (start, mid, end) for instant visual QC."""
    offsets = [0.2, max(0.2, dur * 0.5), max(0.2, dur - 0.3)]
    rows = []
    cap_c = cv2.VideoCapture(cen_path)
    cap_u = cv2.VideoCapture(unc_path)

    for off in offsets:
        cap_c.set(cv2.CAP_PROP_POS_MSEC, (cen_start + off) * 1000.0)
        cap_u.set(cv2.CAP_PROP_POS_MSEC, (unc_start + off) * 1000.0)
        ok_c, fc = cap_c.read()
        ok_u, fu = cap_u.read()
        if not ok_c or fc is None:
            fc = _grab_frame_ffmpeg(cen_path, cen_start + off)
        else:
            fc = cv2.resize(fc, (480, 270))
        if not ok_u or fu is None:
            fu = _grab_frame_ffmpeg(unc_path, unc_start + off)
        else:
            fu = cv2.resize(fu, (480, 270))
        if fc is not None and fu is not None:
            cv2.putText(fc, f"CEN {fmt_tc(cen_start + off)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(fu, f"UNC {fmt_tc(unc_start + off)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            rows.append(np.hstack([fc, fu]))

    cap_c.release()
    cap_u.release()
    if rows:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        cv2.imwrite(out_png, np.vstack(rows))


def export_silent_clip(src_path, dst_path, start_sec, duration_sec, target_fps):
    """Cut a continuous segment with audio stripped (-an) and frame rate locked using GPU NVENC."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    nvenc_cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start_sec:.6f}",
        "-i", src_path,
        "-t", f"{duration_sec:.6f}",
        "-vf", f"fps={target_fps:.6f}",
        "-an",
        "-c:v", "h264_nvenc", "-cq", "17", "-preset", "p4",
        dst_path,
    ]
    res = subprocess.run(nvenc_cmd)
    if res.returncode != 0:
        cpu_cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{start_sec:.6f}",
            "-i", src_path,
            "-t", f"{duration_sec:.6f}",
            "-vf", f"fps={target_fps:.6f}",
            "-an",
            "-c:v", "libx264", "-crf", "16", "-preset", "fast",
            dst_path,
        ]
        subprocess.run(cpu_cmd, check=True)


def merge_contiguous_windows(raw_hits, max_offset_drift=0.18, max_bridge_gap=1.0):
    """Merge matched windows that stay on the same uncensored file and exact time offset.
    Bridges brief (<= 3.0s) low-contrast close-ups ONLY when the time offset before and after
    is identical (same continuous take), while splitting real cuts/alternate takes."""
    if not raw_hits:
        return []
    merged = [dict(raw_hits.pop(0))]
    for hit in raw_hits:
        prev = merged[-1]
        same_unc_file = (prev["unc_path"] == hit["unc_path"])
        prev_offset = prev["unc_start"] - prev["cen_start"]
        cur_offset = hit["unc_start"] - hit["cen_start"]
        prev_end = prev["cen_start"] + prev["duration"]
        gap = hit["cen_start"] - prev_end

        if same_unc_file and gap <= max_bridge_gap and abs(cur_offset - prev_offset) <= max_offset_drift:
            new_end = max(prev_end, hit["cen_start"] + hit["duration"])
            prev["duration"] = round(new_end - prev["cen_start"], 3)
            prev["corr"] = round(min(prev["corr"], hit["corr"]), 4)
        else:
            merged.append(dict(hit))
    return merged

def refine_exact_offset(cen_path, unc_path, cen_start, unc_start, search_sec=0.35, probe_offset=1.0):
    """Refine unc_start to exact native-frame precision (eliminates 6fps ±4 frame quantization)."""
    probe_c = cen_start + probe_offset
    probe_u = max(0.0, unc_start + probe_offset - search_sec)

    cap_c = cv2.VideoCapture(cen_path)
    fps_c = cap_c.get(cv2.CAP_PROP_FPS) or 24.0
    cap_c.set(cv2.CAP_PROP_POS_MSEC, probe_c * 1000.0)
    c_seq = []
    for _ in range(10):
        ok, f = cap_c.read()
        if not ok or f is None:
            break
        gray = cv2.cvtColor(cv2.resize(f, (128, 72), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        c_seq.append(extract_robust_fingerprint(gray, w=128, h=72))
    cap_c.release()

    cap_u = cv2.VideoCapture(unc_path)
    fps_u = cap_u.get(cv2.CAP_PROP_FPS) or 24.0
    cap_u.set(cv2.CAP_PROP_POS_MSEC, probe_u * 1000.0)
    n_u_read = int(round((2.0 * search_sec + 0.5) * fps_u))
    u_seq = []
    for _ in range(max(20, n_u_read)):
        ok, f = cap_u.read()
        if not ok or f is None:
            break
        gray = cv2.cvtColor(cv2.resize(f, (128, 72), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        u_seq.append(extract_robust_fingerprint(gray, w=128, h=72))
    cap_u.release()

    if len(c_seq) < 6 or len(u_seq) <= len(c_seq):
        return unc_start

    c_arr = np.vstack(c_seq)
    u_arr = np.vstack(u_seq)
    n_dim = len(next(iter(c_arr)))
    best_idx = 0
    best_score = -1e9
    for shift in range(len(u_arr) - len(c_arr) + 1):
        score = float(np.sum(u_arr[shift:shift + len(c_arr)] * c_arr)) / (len(c_arr) * n_dim)
        if score > best_score:
            best_score = score
            best_idx = shift

    matched_u_time = probe_u + (best_idx / fps_u)
    refined_unc_start = max(0.0, matched_u_time - probe_offset)
    return round(refined_unc_start, 4)

def match_group(group_label, cen_paths, unc_paths, qc_dir, export_dir=None, chunk_sec=5.0, min_corr=0.78, target_fps=6.0):
    """Fast coarse-to-fine scene matching across all censored vs uncensored files in a group."""
    print(f"\n=== Group: {group_label} ({len(cen_paths)} censored vs {len(unc_paths)} uncensored) ===")
    cache_dir = os.path.join(qc_dir, "_index_cache")
    unc_indexed = []
    for u_path in unc_paths:
        u_mat, u_ts, u_fps = index_video_ffmpeg(u_path, cache_dir=cache_dir, target_fps=target_fps)
        if len(u_mat) > 0:
            unc_indexed.append((u_path, u_mat, u_ts, u_fps))

    if not unc_indexed:
        return []

    win = max(12, int(round(chunk_sec * target_fps)))
    stride = max(6, win // 2)
    mid_k = win // 2
    sample_ks = list(range(0, win, 2))
    all_group_scenes = []

    for c_path in cen_paths:
        c_mat, c_ts, c_fps = index_video_ffmpeg(c_path, cache_dir=cache_dir, target_fps=target_fps)
        if len(c_mat) < win:
            continue

        n_dim = len(next(iter(c_mat)))
        raw_hits = []
        search_starts = list(range(0, len(c_mat) - win + 1, stride))

        for c_start_i in tqdm(search_starts, desc=f"  matching {os.path.basename(c_path)}", unit="win"):
            query = c_mat[c_start_i:c_start_i + win]
            first_frame = next(iter(query))
            last_frame = query[-1]
            if (
                float(np.mean(np.std(query, axis=1))) < 0.2
                or float(np.std(first_frame)) < 0.25
                or float(np.std(last_frame)) < 0.25
            ):
                continue

            q_a, q_b, q_c = query[2], query[mid_k], query[win - 3]
            best_overall = None

            for u_path, u_mat, u_ts, u_fps in unc_indexed:
                n_positions = len(u_mat) - win + 1
                if n_positions <= 0:
                    continue

                 # Stage 1: 3-anchor 1D matrix-vector dot product (immune to single-frame flashes)
                coarse = (
                    np.dot(u_mat[2:2 + n_positions], q_a)
                    + np.dot(u_mat[mid_k:mid_k + n_positions], q_b)
                    + np.dot(u_mat[win - 3:win - 3 + n_positions], q_c)
                ) / (3.0 * n_dim)
                top_k = min(40, n_positions)
                cand_indices = np.argpartition(coarse, -top_k)[-top_k:]

                # Stage 2: Full multi-frame window verification on top 25 candidates only
                for u_i_raw in cand_indices:
                    u_i = int(u_i_raw)
                    corr = float(np.mean([np.dot(u_mat[u_i + k], query[k]) for k in sample_ks])) / n_dim
                    if best_overall is None or corr > next(iter(best_overall)):
                        start_c = float(np.dot(u_mat[u_i], next(iter(query)))) / n_dim
                        end_c = float(np.dot(u_mat[u_i + win - 1], query[-1])) / n_dim
                        best_overall = (corr, start_c, end_c, u_path, float(u_ts[u_i]), u_fps)

            if best_overall is not None:
                corr, start_c, end_c, u_path, u_t, u_fps = best_overall
                if corr >= min_corr and start_c >= min_corr and end_c >= min_corr:
                    raw_hits.append({
                        "cen_path": c_path,
                        "unc_path": u_path,
                        "cen_start": round(float(c_ts[c_start_i]), 3),
                        "unc_start": round(u_t, 3),
                        "duration": chunk_sec,
                        "corr": round(corr, 4),
                        "locked_fps": min(c_fps, u_fps),
                    })

        scenes = merge_contiguous_windows(raw_hits)
        c_stem, _ = os.path.splitext(os.path.basename(c_path))
        print(f"  {os.path.basename(c_path)} -> {len(scenes)} continuous matched scene(s):")

        iterator = tqdm(scenes, desc="  exporting clips", unit="clip") if export_dir else scenes
        for s_idx, sc in enumerate(iterator):
            if sc["duration"] >= 2.0:
                sc["unc_start"] = refine_exact_offset(sc["cen_path"], sc["unc_path"], sc["cen_start"], sc["unc_start"])
            base_tag = group_label if group_label.lower() == c_stem.lower() else f"{group_label}_{c_stem}"
            clip_name = f"{base_tag}_s{s_idx:03d}.mp4"
            qc_png = os.path.join(qc_dir, f"{base_tag}_s{s_idx:03d}_qc.jpg")
            save_qc_strip(sc["cen_path"], sc["unc_path"], sc["cen_start"], sc["unc_start"], sc["duration"], qc_png)

            tqdm.write(
                f"    s{s_idx:03d}: {os.path.basename(sc['cen_path'])} [{fmt_tc(sc['cen_start'])}] <-> "
                f"{os.path.basename(sc['unc_path'])} [{fmt_tc(sc['unc_start'])}] "
                f"(dur={sc['duration']:.1f}s, corr={sc['corr']:.3f})"
            )
            if export_dir:
                cen_out = os.path.join(export_dir, "censored", clip_name)
                unc_out = os.path.join(export_dir, "uncensored", clip_name)
                export_silent_clip(sc["cen_path"], cen_out, sc["cen_start"], sc["duration"], sc["locked_fps"])
                export_silent_clip(sc["unc_path"], unc_out, sc["unc_start"], sc["duration"], sc["locked_fps"])

            all_group_scenes.append({
                "clip_name": clip_name,
                "cen_source": os.path.basename(sc["cen_path"]),
                "unc_source": os.path.basename(sc["unc_path"]),
                "cen_start": sc["cen_start"],
                "unc_start": sc["unc_start"],
                "duration": sc["duration"],
                "corr": sc["corr"],
                "qc_image": qc_png,
            })

    return all_group_scenes


def discover_groups(censored_dir, uncensored_dir):
    """Discover both matching subfolders (N-vs-M) and matching root filename stems (1-to-1)."""
    groups = []

    c_subdirs = {d for d in os.listdir(censored_dir) if os.path.isdir(os.path.join(censored_dir, d))}
    u_subdirs = {d for d in os.listdir(uncensored_dir) if os.path.isdir(os.path.join(uncensored_dir, d))}
    for folder in sorted(c_subdirs & u_subdirs):
        c_vids = sorted(
            os.path.join(censored_dir, folder, f)
            for f in os.listdir(os.path.join(censored_dir, folder))
            if f.lower().endswith(VIDEO_EXTS)
        )
        u_vids = sorted(
            os.path.join(uncensored_dir, folder, f)
            for f in os.listdir(os.path.join(uncensored_dir, folder))
            if f.lower().endswith(VIDEO_EXTS)
        )
        if c_vids and u_vids:
            groups.append((folder, c_vids, u_vids))

    c_root = {
        os.path.splitext(f)[0].lower(): os.path.join(censored_dir, f)
        for f in os.listdir(censored_dir)
        if os.path.isfile(os.path.join(censored_dir, f)) and f.lower().endswith(VIDEO_EXTS)
    }
    u_root = {
        os.path.splitext(f)[0].lower(): os.path.join(uncensored_dir, f)
        for f in os.listdir(uncensored_dir)
        if os.path.isfile(os.path.join(uncensored_dir, f)) and f.lower().endswith(VIDEO_EXTS)
    }
    for stem_key in sorted(c_root.keys() & u_root.keys()):
        label = os.path.splitext(os.path.basename(c_root[stem_key]))[0]
        groups.append((label, [c_root[stem_key]], [u_root[stem_key]]))

    return groups

def _grab_frame_ffmpeg(video_path, ts_sec, w=480, h=270):
    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", f"{ts_sec:.3f}",
        "-i", video_path,
        "-frames:v", "1",
        "-vf", f"scale={w}:{h}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if len(res.stdout) == w * h * 3:
        return np.frombuffer(res.stdout, dtype=np.uint8).reshape((h, w, 3)).copy()
    return None

def main():
    parser = argparse.ArgumentParser(description="Match reordered/multi-part scenes via subfolder grouping or root 1-to-1 filenames.")
    parser.add_argument("--censored_dir", default=os.path.join("datasets", "raw", "censored"))
    parser.add_argument("--uncensored_dir", default=os.path.join("datasets", "raw", "uncensored"))
    parser.add_argument("--qc_dir", default=os.path.join("datasets", "raw", "sync_qc"))
    parser.add_argument("--export_dir", default="", help="If set, exports silent (-an) FPS-locked .mp4 pairs")
    parser.add_argument("--chunk_sec", type=float, default=5.0, help="Sliding window size (seconds)")
    parser.add_argument("--min_corr", type=float, default=0.78, help="Minimum cosine similarity (0..1)")
    args = parser.parse_args()

    groups = discover_groups(args.censored_dir, args.uncensored_dir)
    if not groups:
        print("No matching subfolders or root video filenames found.")
        return

    os.makedirs(args.qc_dir, exist_ok=True)
    for label, c_paths, u_paths in groups:
        group_scenes = match_group(
            label, c_paths, u_paths, args.qc_dir,
            export_dir=args.export_dir or None,
            chunk_sec=args.chunk_sec,
            min_corr=args.min_corr,
        )
        group_report_path = os.path.join(args.qc_dir, f"{label}_scene_match_report.json")
        with open(group_report_path, "w", encoding="utf-8") as f:
            json.dump({label: group_scenes}, f, indent=2)
        print(f"  Saved group match report -> {group_report_path}")

    print(f"\nInspect side-by-side QC images in -> {args.qc_dir}")
    if args.export_dir:
        print(f"Exported silent aligned pairs -> {args.export_dir}/censored & {args.export_dir}/uncensored")
        cache_dir = os.path.join(args.qc_dir, "_index_cache")
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"Cleaned up temporary index cache -> {cache_dir}")


if __name__ == "__main__":
    main()