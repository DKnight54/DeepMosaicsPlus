import os
import re
import json
import shutil
import datetime
from operator import itemgetter


def deploy_all(manifest_path=None, checkpoints_root=None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if manifest_path is None:
        manifest_path = os.path.join(root, "pretrained_models", "mosaic", "size_manifest.json")
    if checkpoints_root is None:
        checkpoints_root = os.path.join(root, "checkpoints")

    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    man_base = os.path.splitext(os.path.basename(manifest_path))[0]
    ckpt_prefix = "" if man_base == "size_manifest" else man_base

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    deployed = 0

    for pct_key, entry in sorted(manifest.items(), key=lambda kv: float(next(iter(kv)))):
        rel_target = entry.get("model", "") if isinstance(entry, dict) else str(entry)
        if not rel_target:
            print(f"[skip] Bucket {pct_key}%: no 'model' path in manifest entry")
            continue
        sub_name = os.path.join(ckpt_prefix, f"clean_{pct_key}pct") if ckpt_prefix else f"clean_{pct_key}pct"
        ckpt_dir = os.path.join(checkpoints_root, sub_name)
        if not os.path.isdir(ckpt_dir):
            print(f"[skip] Bucket {pct_key}%: checkpoint folder not found ({ckpt_dir})")
            continue

        candidates = []
        for fname in os.listdir(ckpt_dir):
            m = re.fullmatch(r"(\d+)_G\.pth", fname)
            if m:
                candidates.append((int(m.group(1)), fname))

        if not candidates:
            print(f"[skip] Bucket {pct_key}%: no *_G.pth checkpoints found in {ckpt_dir}")
            continue

        latest_iter, latest_pth = max(candidates, key=itemgetter(0))
        src_pth = os.path.join(ckpt_dir, latest_pth)
        src_stem, _ = os.path.splitext(src_pth)
        src_json = src_stem + ".json"

        if os.path.isabs(rel_target):
            dst_pth = rel_target
        elif rel_target.replace("\\", "/").startswith("pretrained_models/"):
            dst_pth = os.path.join(root, rel_target)
        else:
            dst_pth = os.path.join(os.path.dirname(manifest_path), rel_target)

        dst_stem, _ = os.path.splitext(dst_pth)
        dst_json = dst_stem + ".json"
        os.makedirs(os.path.dirname(dst_pth), exist_ok=True)

        for dst_file in (dst_pth, dst_json):
            if os.path.isfile(dst_file):
                base, ext = os.path.splitext(dst_file)
                backup_path = f"{base}_bak_{ts}{ext}"
                os.replace(dst_file, backup_path)
                print(f"[backup] {os.path.basename(dst_file)} -> {os.path.basename(backup_path)}")

        shutil.copy2(src_pth, dst_pth)
        if os.path.isfile(src_json):
            shutil.copy2(src_json, dst_json)
            print(f"[deploy] Bucket {pct_key}%: {latest_pth} (+ .json) -> {os.path.basename(dst_pth)}")
        else:
            print(f"[deploy] Bucket {pct_key}%: {latest_pth} (WARNING: no .json found) -> {os.path.basename(dst_pth)}")

        deployed += 1

    print(f"\nDone! Deployed {deployed} bucket model(s) into {os.path.dirname(manifest_path)}.")
    return deployed


if __name__ == "__main__":
    deploy_all()