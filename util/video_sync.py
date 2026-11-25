import os
import json
import subprocess
import cv2
import numpy as np

STD_RATES = (24000 / 1001, 24.0, 25.0, 30000 / 1001, 30.0, 48.0, 50.0, 60000 / 1001, 60.0)
VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".ts", ".m4v", ".webm")


def extract_robust_fingerprint(gray_hw, w=128, h=72):
    roi = gray_hw[int(h * 0.12):int(h * 0.74), int(w * 0.04):int(w * 0.96)]
    coarse_center = cv2.resize(roi, (48, 27), interpolation=cv2.INTER_AREA)
    y1, y2 = int(h * 0.16), int(h * 0.72)
    left_strip = gray_hw[y1:y2, int(w * 0.04):int(w * 0.22)]
    right_strip = gray_hw[y1:y2, int(w * 0.78):int(w * 0.96)]
    feat = np.concatenate([coarse_center.ravel() * 1.5, left_strip.ravel(), right_strip.ravel()])
    std = float(np.std(feat))
    return (feat - float(np.mean(feat))) / std if std > 1e-3 else np.zeros_like(feat)


def index_video_ffmpeg(video_path, cache_dir=None, target_fps=6.0, w=128, h=72):
    st = os.stat(video_path)
    cache_file = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{os.path.basename(video_path)}_{st.st_size}_{int(target_fps)}fps.npz")
        if os.path.isfile(cache_file):
            d = np.load(cache_file)
            return d["mat"], d["ts"], float(d["src_fps"])

    src_fps = probe_cfr_fps(video_path, halve_high_fps=False)
    cmd = ["ffmpeg", "-v", "error", "-i", video_path, "-vf", f"fps={target_fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = w * h
    feats, timestamps, idx = [], [], 0
    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        gray = np.frombuffer(raw, dtype=np.uint8).reshape((h, w)).astype(np.float32)
        feats.append(extract_robust_fingerprint(gray, w=w, h=h))
        timestamps.append(idx / target_fps)
        idx += 1
    proc.stdout.close()
    proc.wait()

    mat = np.ascontiguousarray(np.vstack(feats), dtype=np.float32) if feats else np.empty((0, 1), dtype=np.float32)
    ts_arr = np.array(timestamps, dtype=np.float64)
    if cache_file and len(mat) > 0:
        np.savez(cache_file, mat=mat, ts=ts_arr, src_fps=np.array(src_fps))
    return mat, ts_arr, src_fps

def merge_contiguous_windows(raw_hits, max_offset_drift=0.18, max_bridge_gap=1.0):
    if not raw_hits:
        return []
    merged = [dict(raw_hits[0])]
    for hit in raw_hits[1:]:
        prev = merged[-1]
        same_u = (prev["unc_path"] == hit["unc_path"])
        prev_off = prev["unc_start"] - prev["cen_start"]
        cur_off = hit["unc_start"] - hit["cen_start"]
        prev_end = prev["cen_start"] + prev["duration"]
        if same_u and (hit["cen_start"] - prev_end) <= max_bridge_gap and abs(cur_off - prev_off) <= max_offset_drift:
            prev["duration"] = round(max(prev_end, hit["cen_start"] + hit["duration"]) - prev["cen_start"], 3)
            prev["corr"] = round(min(prev["corr"], hit["corr"]), 4)
        else:
            merged.append(dict(hit))
    return merged

def refine_exact_offset(cen_path, unc_path, cen_start, unc_start, search_sec=0.35, probe_offset=1.0):
    probe_c = cen_start + probe_offset
    probe_u = max(0.0, unc_start + probe_offset - search_sec)
    cap_c = cv2.VideoCapture(cen_path)
    cap_c.set(cv2.CAP_PROP_POS_MSEC, probe_c * 1000.0)
    c_seq = []
    for _ in range(10):
        ok, f = cap_c.read()
        if not ok or f is None:
            break
        c_seq.append(extract_robust_fingerprint(cv2.cvtColor(cv2.resize(f, (128, 72)), cv2.COLOR_BGR2GRAY).astype(np.float32)))
    cap_c.release()

    cap_u = cv2.VideoCapture(unc_path)
    fps_u = probe_cfr_fps(unc_path, halve_high_fps=False)
    cap_u.set(cv2.CAP_PROP_POS_MSEC, probe_u * 1000.0)
    u_seq = []
    for _ in range(max(20, int(round((2.0 * search_sec + 0.5) * fps_u)))):
        ok, f = cap_u.read()
        if not ok or f is None:
            break
        u_seq.append(extract_robust_fingerprint(cv2.cvtColor(cv2.resize(f, (128, 72)), cv2.COLOR_BGR2GRAY).astype(np.float32)))
    cap_u.release()

    if len(c_seq) < 6 or len(u_seq) <= len(c_seq):
        return unc_start
    c_arr, u_arr = np.vstack(c_seq), np.vstack(u_seq)
    n_dim = c_arr.shape[1]
    best_i = max(range(len(u_arr) - len(c_arr) + 1), key=lambda s: float(np.sum(u_arr[s:s + len(c_arr)] * c_arr)) / (len(c_arr) * n_dim))
    return round(max(0.0, probe_u + (best_i / fps_u) - probe_offset), 4)

def discover_groups(censored_dir, uncensored_dir):
    groups = []
    c_sub = {d for d in os.listdir(censored_dir) if os.path.isdir(os.path.join(censored_dir, d))}
    u_sub = {d for d in os.listdir(uncensored_dir) if os.path.isdir(os.path.join(uncensored_dir, d))}
    for folder in sorted(c_sub & u_sub):
        cv_list = sorted(os.path.join(censored_dir, folder, f) for f in os.listdir(os.path.join(censored_dir, folder)) if f.lower().endswith(VIDEO_EXTS))
        uv_list = sorted(os.path.join(uncensored_dir, folder, f) for f in os.listdir(os.path.join(uncensored_dir, folder)) if f.lower().endswith(VIDEO_EXTS))
        if cv_list and uv_list:
            groups.append((folder, cv_list, uv_list))
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
    for k in sorted(c_root.keys() & u_root.keys()):
        stem = os.path.splitext(os.path.basename(c_root[k]))[0]
        groups.append((stem, [c_root[k]], [u_root[k]]))
    return groups


def match_group(group_label, cen_paths, unc_paths, qc_dir, export_dir=None, chunk_sec=5.0, min_corr=0.78, target_fps=6.0):
    cache_dir = os.path.join(qc_dir, "_index_cache")
    unc_indexed = []
    for u in unc_paths:
        u_mat, u_ts, u_fps = index_video_ffmpeg(u, cache_dir=cache_dir, target_fps=target_fps)
        if len(u_mat) > 0:
            unc_indexed.append((u, u_mat, u_ts, u_fps))
    if not unc_indexed:
        return []
    win = max(12, int(round(chunk_sec * target_fps)))
    stride, mid_k, sample_ks = max(6, win // 2), win // 2, list(range(0, win, 2))
    out_scenes = []
    for c_path in cen_paths:
        c_mat, c_ts, c_fps = index_video_ffmpeg(c_path, cache_dir=cache_dir, target_fps=target_fps)
        if len(c_mat) < win:
            continue
        n_dim = int(c_mat.shape[1])
        raw_hits = []
        for cs in range(0, len(c_mat) - win + 1, stride):
            q = c_mat[cs:cs + win]
            if float(np.mean(np.std(q, axis=1))) < 0.2 or float(np.std(q[0])) < 0.25 or float(np.std(q[-1])) < 0.25:
                continue
            qa, qb, qc = q[2], q[mid_k], q[win - 3]
            best = None
            for u_path, u_mat, u_ts, u_fps in unc_indexed:
                npos = len(u_mat) - win + 1
                if npos <= 0:
                    continue
                coarse = (
                    np.dot(u_mat[2:2 + npos], qa)
                    + np.dot(u_mat[mid_k:mid_k + npos], qb)
                    + np.dot(u_mat[win - 3:win - 3 + npos], qc)
                ) / (3.0 * n_dim)
                top_k = min(40, npos)
                for ui_raw in np.argpartition(coarse, -top_k)[-top_k:]:
                    ui = int(ui_raw)
                    corr = float(np.mean([np.dot(u_mat[ui + k], q[k]) for k in sample_ks])) / n_dim
                    if best is None or corr > best[0]:
                        sc = float(np.dot(u_mat[ui], q[0])) / n_dim
                        ec = float(np.dot(u_mat[ui + win - 1], q[-1])) / n_dim
                        best = (corr, sc, ec, u_path, float(u_ts[ui]), u_fps)
            if best is not None and best[0] >= min_corr and best[1] >= min_corr and best[2] >= min_corr:
                raw_hits.append({
                    "cen_path": c_path,
                    "unc_path": best[3],
                    "cen_start": round(float(c_ts[cs]), 3),
                    "unc_start": round(best[4], 3),
                    "duration": chunk_sec,
                    "corr": round(best[0], 4),
                })
        scenes = merge_contiguous_windows(raw_hits)
        c_stem = os.path.splitext(os.path.basename(c_path))[0]
        for s_idx, sc in enumerate(scenes):
            if sc["duration"] >= 2.0:
                sc["unc_start"] = refine_exact_offset(sc["cen_path"], sc["unc_path"], sc["cen_start"], sc["unc_start"])
            base_tag = group_label if group_label.lower() == c_stem.lower() else f"{group_label}_{c_stem}"
            out_scenes.append({
                "clip_name": f"{base_tag}_s{s_idx:03d}.mp4",
                "cen_source": os.path.basename(sc["cen_path"]),
                "unc_source": os.path.basename(sc["unc_path"]),
                "cen_start": sc["cen_start"],
                "unc_start": sc["unc_start"],
                "duration": sc["duration"],
                "corr": sc["corr"],
            })
    return out_scenes

def _parse_frac(rate_str):
    if not rate_str or rate_str in ("0/0", "N/A"):
        return 0.0
    if "/" in rate_str:
        num, den = rate_str.split("/", 1)
        d = float(den)
        return float(num) / d if d != 0 else 0.0
    return float(rate_str)


def probe_cfr_fps(video_path, halve_high_fps=True):
    """Return clean target CFR float for a video using ffprobe avg_frame_rate + r_frame_rate."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate",
        "-of", "json", video_path,
    ]
    fps = 0.0
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        streams = json.loads(res.stdout or "{}").get("streams") or [{}]
        info = next(iter(streams))
        avg_f = _parse_frac(info.get("avg_frame_rate", "0/0"))
        r_f = _parse_frac(info.get("r_frame_rate", "0/0"))
        # Prefer avg_frame_rate when r_frame_rate is inflated (e.g. VFR 1000fps or 29.97 pulldown over 23.976)
        fps = avg_f if (10.0 <= avg_f <= 120.0) else r_f
    except Exception:
        pass

    if fps <= 0.0:
        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        cap.release()

    # Snap within 1.5% to exact broadcast/film rational rate (23.976, 24, 25, 29.97, 30, 50, 59.94, 60)
    for std in STD_RATES:
        if abs(fps - std) / std < 0.015:
            fps = std
            break

    if halve_high_fps and fps > 45.0:
        fps *= 0.5
    return round(fps, 6)


def resolve_pair_cfr(cen_path, unc_path, halve_high_fps=True):
    """Pick the lower native CFR across a pair so neither stream ever duplicates frames."""
    fc = probe_cfr_fps(cen_path, halve_high_fps=halve_high_fps)
    fu = probe_cfr_fps(unc_path, halve_high_fps=halve_high_fps)
    return min(fc, fu)


def _border_zvec(bgr, size=144, border_ratio=0.15):
    small = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = max(4, int(round(size * border_ratio)))
    ring = np.concatenate([
        gray[:b, :].ravel(),
        gray[size - b:, :].ravel(),
        gray[b:size - b, :b].ravel(),
        gray[b:size - b, size - b:].ravel(),
    ])
    std = float(np.std(ring))
    if std < 2.0:
        return None
    return (ring - float(np.mean(ring))) / std


def align_pair_folders_dtw(orig_dir, mos_dir, max_band=3, min_mean_corr=0.80):
    """Monotonic Banded DTW alignment on full-frame extracted JPGs before ROI cropping.
    Repairs both constant +/-1..3f offsets AND mid-clip VFR/pulldown slips in-place.
    Returns (ok: bool, mean_corr: float, info_str: str).
    """
    o_names = sorted(f for f in os.listdir(orig_dir) if f.lower().endswith(".jpg"))
    m_names = sorted(f for f in os.listdir(mos_dir) if f.lower().endswith(".jpg"))
    no, nm = len(o_names), len(m_names)
    if no < 20 or nm < 20:
        return False, 0.0, "too_few_frames"

    o_vecs = [_border_zvec(cv2.imread(os.path.join(orig_dir, f))) for f in o_names]
    m_vecs = [_border_zvec(cv2.imread(os.path.join(mos_dir, f))) for f in m_names]

    # Build banded similarity matrix S[i, j] for |i - j| <= max_band
    NEG = -1e9
    dp = np.full((no, nm), NEG, dtype=np.float64)
    back = np.zeros((no, nm, 2), dtype=np.int16)

    def sim(i, j):
        ov, mv = o_vecs[i], m_vecs[j]
        if ov is None or mv is None:
            return 0.85
        return float(np.mean(ov * mv))

    for i in range(min(no, max_band + 1)):
        dp[i, 0] = sim(i, 0) - 0.002 * i
        back[i, 0] = (-1, -1)
    for j in range(1, min(nm, max_band + 1)):
        dp[0, j] = sim(0, j) - 0.002 * j
        back[0, j] = (-1, -1)

    for i in range(1, no):
        j_lo = max(1, i - max_band)
        j_hi = min(nm, i + max_band + 1)
        for j in range(j_lo, j_hi):
            s_ij = sim(i, j)
            # Transitions: (1,1) normal step, (2,1) orig had extra frame, (1,2) mos had extra frame
            cands = [
                (dp[i - 1, j - 1] + s_ij, i - 1, j - 1),
                (dp[i - 2, j - 1] + s_ij - 0.005 if i >= 2 else NEG, i - 2, j - 1),
                (dp[i - 1, j - 2] + s_ij - 0.005 if j >= 2 else NEG, i - 1, j - 2),
            ]
            best_val, bi, bj = max(cands, key=lambda x: x[0])
            dp[i, j] = best_val
            back[i, j] = (bi, bj)

    # Pick best end within max_band of final frames
    best_end = None
    best_score = NEG
    for i in range(max(0, no - max_band - 1), no):
        for j in range(max(0, nm - max_band - 1), nm):
            if abs(i - j) <= max_band and dp[i, j] > best_score:
                best_score = dp[i, j]
                best_end = (i, j)

    if best_end is None:
        return False, 0.0, "no_valid_dtw_path"

    path = []
    ci, cj = best_end
    while ci >= 0 and cj >= 0:
        path.append((ci, cj))
        pi, pj = int(back[ci, cj, 0]), int(back[ci, cj, 1])
        ci, cj = pi, pj
    path.reverse()

    corrs = [sim(i, j) for i, j in path if o_vecs[i] is not None and m_vecs[j] is not None]
    mean_corr = float(np.mean(corrs)) if corrs else 0.0
    shifts = [i - j for i, j in path]
    if mean_corr < min_mean_corr:
        return False, mean_corr, f"low_corr={mean_corr:.3f}"

    # Rewrite aligned sequence atomically via temp staging
    tmp_o = orig_dir + "_dtw"
    tmp_m = mos_dir + "_dtw"
    os.makedirs(tmp_o, exist_ok=True)
    os.makedirs(tmp_m, exist_ok=True)
    for out_k, (oi, mj) in enumerate(path, start=1):
        os.link(os.path.join(orig_dir, o_names[oi]), os.path.join(tmp_o, f"{out_k:05d}.jpg"))
        os.link(os.path.join(mos_dir, m_names[mj]), os.path.join(tmp_m, f"{out_k:05d}.jpg"))

    for f in os.listdir(orig_dir):
        os.remove(os.path.join(orig_dir, f))
    for f in os.listdir(mos_dir):
        os.remove(os.path.join(mos_dir, f))
    for f in os.listdir(tmp_o):
        os.replace(os.path.join(tmp_o, f), os.path.join(orig_dir, f))
    for f in os.listdir(tmp_m):
        os.replace(os.path.join(tmp_m, f), os.path.join(mos_dir, f))
    os.rmdir(tmp_o)
    os.rmdir(tmp_m)

    return True, mean_corr, f"dtw_corr={mean_corr:.3f} shift_range=[{min(shifts):+d}..{max(shifts):+d}f]"
