"""
Train one clean model per size bucket in your manifest, combining:
  - a shared pool of synthetic origin_image/+mask/ video folders (same data
    reused for every bucket — see docs/training_with_your_own_dataset.md /
    make_datasets/make_video_dataset.py for how to build this), with
    train/clean/train.py's --dataset_mosaic_pct constraining synthesis to
    each bucket's size band at train time, and
  - that bucket's real censored/uncensored pairs, if any (from
    tools/bucket_real_pairs.py's output).

For each bucket, this builds one merged dataset view (via symlinks/junctions
so the — possibly large — synthetic frame data isn't duplicated on disk),
then runs train/clean/train.py against it with --dataset_mosaic_pct set,
looping through every bucket in your manifest sequentially.

This has NOT been run end-to-end (no GPU/torch in the environment this was
built in) — recommend a quick dry run on one bucket with a tiny synthetic
pool and --n_epoch 1 before committing to a full multi-bucket run.

Usage:
    python tools/train_all_buckets.py \
        --synthetic_dataset datasets/synthetic_pool \
        --real_pairs_dataset datasets/real_pairs \
        --model_manifest pretrained_models/mosaic/size_manifest.json \
        --work_dir datasets/_merged \
        -- --batchsize 16 --n_epoch 200 --gpu_id 0

Anything after a bare `--` is forwarded as-is to train/clean/train.py
(finesize, batchsize, n_epoch, gpu_id, lr, etc.) — this script only injects
--dataset, --dataset_mosaic_pct, and --savename itself.
"""
import os
import sys
import shutil
import platform
import subprocess
import argparse
import codecs
import time
import json

# Windows consoles often default to a non-UTF-8 code page (cp1252/cp437),
# which mangles Unicode characters like em-dashes into '?' or '�'. Force
# UTF-8 stdout so any character in this script's output prints correctly
# regardless of the system's console code page.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from util import diagnose
from models.loadmodel import load_size_manifest_details


def link_or_copy(src, dst):
    """Create dst -> src as a symlink (Unix/macOS/Windows dev mode), falling
    back to an NTFS directory junction (Windows, no special privileges),
    falling back to a full copy (always works, costs disk + time) as a last
    resort. Skips silently if dst already exists (idempotent re-runs).
    """
    if os.path.exists(dst):
        return
    src = os.path.abspath(src)
    try:
        os.symlink(src, dst, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    if platform.system() == 'Windows':
        try:
            subprocess.run(['cmd', '/c', 'mklink', '/J', dst, src],
                            check=True, capture_output=True)
            return
        except Exception:
            pass
    print(f"  (falling back to copying {os.path.basename(src)} — symlink/junction unavailable; "
          f"this uses extra disk space and time)")
    shutil.copytree(src, dst)


def build_merged_dataset(work_dir, bucket_label, synthetic_dataset, real_pairs_dataset):
    merged_dir = os.path.join(work_dir, f'bucket_{bucket_label}_merged')
    os.makedirs(merged_dir, exist_ok=True)

    n_synth = 0
    if synthetic_dataset and os.path.isdir(synthetic_dataset):
        for name in os.listdir(synthetic_dataset):
            src = os.path.join(synthetic_dataset, name)
            if os.path.isdir(src):
                link_or_copy(src, os.path.join(merged_dir, name))
                n_synth += 1

    n_real = 0
    if real_pairs_dataset:
        bucket_dir = os.path.join(real_pairs_dataset, f'bucket_{bucket_label}')
        if os.path.isdir(bucket_dir):
            for name in os.listdir(bucket_dir):
                src = os.path.join(bucket_dir, name)
                if os.path.isdir(src):
                    # prefix to avoid name collisions with synthetic folders
                    link_or_copy(src, os.path.join(merged_dir, f'realpair_{name}'))
                    n_real += 1

    return merged_dir, n_synth, n_real


def bucket_finished(savename):
    """(finished?, description). A bucket is finished when train.py wrote checkpoints/<savename>/completed.json.
    Buckets trained before that marker existed are recognised from train_log.txt: the last run in it reached
    its final iteration and did not fail, and a checkpoint file exists."""
    import re
    folder = os.path.join(ROOT, 'checkpoints', savename)
    marker = os.path.join(folder, 'completed.json')
    if os.path.exists(marker):
        try:
            with open(marker) as f:
                info = json.load(f)
            return True, f"finished {info.get('finished', '')}, {info.get('iterations', '?')} iterations at {info.get('finesize', '?')}px"
        except Exception:
            return True, "finished"
    log_path = os.path.join(folder, 'train_log.txt')
    try:
        has_ckpt = any(n.endswith('_G.pth') for n in os.listdir(folder))
        with open(log_path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 600000))
            text = f.read().decode('utf-8', errors='replace').replace('\r', '\n')
    except OSError:
        return False, 'no training log or checkpoint folder yet'
    # Only the last run in the log counts. (A 'Traceback' is NOT a sign of failure: tensorboardX prints a
    # harmless one at every exit. Failures are marked by this script's own 'FAILED' line.)
    run = text.split('\n===== ')[-1]
    if not has_ckpt:
        return False, 'no checkpoint (*_G.pth) saved yet'
    if 'FAILED' in run:
        return False, 'the last run failed'
    if 'Saving final checkpoint' in run or 'Saved ' in run and '_G.pth' in run.split('Saved ')[-1][:40]:
        return True, "the last run saved its final checkpoint (from the log)"
    progress = re.findall(r'(\d+)/(\d+) \[', run)
    if progress and int(progress[-1][1]) > 0 and int(progress[-1][0]) >= int(progress[-1][1]) - 1:
        return True, f"reached iteration {progress[-1][0]} of {progress[-1][1]} (from the log)"
    if progress:
        return False, f"the last run stopped at iteration {progress[-1][0]} of {progress[-1][1]}"
    return False, 'the last run did not get past the start'


def resolve_python_exe():
    """Prefer python.exe over pythonw.exe for launching subprocesses.
    If this script itself was launched via pythonw.exe (e.g. because it was
    started from a .pyw GUI, which Windows associates with pythonw.exe by
    default), sys.executable reports pythonw.exe -- and that choice
    propagates to every subprocess launched with sys.executable, including
    train.py here. pythonw.exe has no console and no real stdio handles,
    which can make output capture through a multi-level subprocess chain
    (GUI -> this script -> train.py) behave inconsistently. python.exe is
    installed alongside pythonw.exe on every standard Windows install, so
    substituting it costs nothing and removes that whole class of doubt.
    """
    exe = sys.executable
    if exe.lower().endswith('pythonw.exe'):
        candidate = exe[:-len('pythonw.exe')] + 'python.exe'
        if os.path.exists(candidate):
            return candidate
    return exe


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--synthetic_dataset', default=None)
    parser.add_argument('--real_pairs_dataset', default=None)
    parser.add_argument('--model_manifest', default=os.path.join('pretrained_models', 'mosaic', 'size_manifest.json'))
    parser.add_argument('--work_dir', default=os.path.join('datasets', '_merged'))
    parser.add_argument('--train_script', default=os.path.join('train', 'clean', 'train.py'))
    parser.add_argument('--buckets', default=None)
    parser.add_argument('--redo_finished', action='store_true')
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    args, train_args = parser.parse_known_args()
    if train_args and train_args[0] == '--':
        train_args = train_args[1:]

    if not os.path.exists(args.model_manifest):
        print(f"--model_manifest not found at {args.model_manifest}")
        sys.exit(1)

    raw_details = load_size_manifest_details(args.model_manifest)
    manifest_details = {f"{k:g}": v for k, v in sorted(raw_details.items())}
    bucket_labels = list(manifest_details.keys())
    if args.buckets:
        wanted = {f"{float(b.strip()):g}" for b in args.buckets.split(',') if b.strip()}
        bucket_labels = [b for b in bucket_labels if b in wanted]

    man_base = os.path.splitext(os.path.basename(args.model_manifest))[0]
    ckpt_prefix = '' if man_base == 'size_manifest' else man_base

    if not args.synthetic_dataset and not args.real_pairs_dataset:
        print("Provide at least one of --synthetic_dataset or --real_pairs_dataset.")
        sys.exit(1)

    print(f"Buckets to train: {bucket_labels}")
    os.makedirs(args.work_dir, exist_ok=True)

    for bucket in bucket_labels:
        info = manifest_details[bucket]
        savename = os.path.join(ckpt_prefix, f'clean_{bucket}pct') if ckpt_prefix else f'clean_{bucket}pct'
        print(f"\n=== Bucket {bucket}% [{info['min_pct']:g}% - {info['max_pct']:g}%] -> {savename} ===")
        done, detail = bucket_finished(savename)
        if done and not args.redo_finished:
            print(f"  Already finished ({detail}); skipping. Tick 'Train finished buckets again' (or pass --redo_finished) to train it again.", flush=True)
            continue
        resume = False
        if not args.redo_finished and not args.no_resume:
            ck = os.path.join(ROOT, 'checkpoints', savename)
            resume = os.path.isdir(ck) and any(n.endswith('_G.pth') for n in os.listdir(ck))
        if detail:
            print(f"  Not finished yet: {detail}. " + ("Resuming from its newest checkpoint." if resume else "Training it from the start."), flush=True)

        merged_dir, n_synth, n_real = build_merged_dataset(
            args.work_dir, bucket, args.synthetic_dataset, args.real_pairs_dataset)
        print(f"  merged dataset: {n_synth} synthetic folder(s) + {n_real} real-pair folder(s) -> {merged_dir}")

        if n_synth == 0 and n_real == 0:
            print("  no data for this bucket at all, skipping.")
            continue

        cmd = [resolve_python_exe(), args.train_script, '--dataset', merged_dir, '--savename', savename]
        if n_synth > 0:
            cmd += [
                '--dataset_mosaic_pct', bucket,
                '--dataset_mosaic_pct_min', str(info['min_pct']),
                '--dataset_mosaic_pct_max', str(info['max_pct']),
            ]
        cmd += train_args
        if resume and '--continue_train' not in cmd:
            cmd += ['--continue_train']

        print(f"  command: {' '.join(cmd)}")
        if args.dry_run:
            continue

        print(f"  system: {diagnose.system_status(ROOT)}", flush=True)
        for warning in diagnose.low_resources(ROOT):
            print(f"  ** WARNING: {warning}", flush=True)
        log_dir = os.path.join(ROOT, 'checkpoints', savename)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'train_log.txt')
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1')
        started = time.time()
        printed = 0
        with open(log_path, 'a', encoding='utf-8', errors='replace') as log:
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(cmd)}\n")
            process = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
            decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
            while True:
                chunk = process.stdout.read1(65536)
                if not chunk:
                    break
                text = decoder.decode(chunk)
                printed += len(text.strip())
                sys.stdout.write(text)
                sys.stdout.flush()
                log.write(text.replace('\r\n', '\n').replace('\r', '\n'))
                log.flush()
            sys.stdout.write(decoder.decode(b'', final=True))
            process.wait()
            code = process.returncode
            if code != 0:
                took = time.time() - started
                msg = (f"\n  ** Training for bucket {bucket}% FAILED after {took:.0f}s. Exit code {diagnose.explain_exit(code)}\n"
                       f"  ** System now: {diagnose.system_status(ROOT)}\n"
                       f"  ** Full output of this run: {log_path}")
                print(msg, flush=True)
                log.write(msg + "\n")
        if code != 0:
            sys.exit(code if 0 < code < 256 else 1)

    print("\nDone. Trained checkpoints are under checkpoints/<savename>/ — use 'Deploy checkpoints to pretrained' to copy/rename each *_G.pth + .json into your manifest's directory.")


if __name__ == '__main__':
    main()