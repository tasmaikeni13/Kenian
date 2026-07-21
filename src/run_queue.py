"""Sequential job runner for one GPU — churns through a queue of training runs.

One long-lived manager per GPU (avoids orphaned CUDA processes from per-job timeouts).
Each job is a dict of train.py CLI args. Completed runs are skipped on restart (their
final.json exists), so the whole queue is resumable.

  CUDA_VISIBLE_DEVICES=0 python src/run_queue.py --queue results/queues/vision_gpu0.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "results")


def run_done(task, tag):
    return os.path.isfile(os.path.join(RESULTS, task, tag, "final.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    args = ap.parse_args()
    jobs = json.load(open(args.queue))
    log = open(args.queue.replace(".json", ".log"), "a")

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    say(
        f"queue {args.queue}: {len(jobs)} jobs; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    for i, job in enumerate(jobs):
        tag = job["tag"]
        task = job["task"]
        if run_done(task, tag):
            say(f"[{i + 1}/{len(jobs)}] SKIP {tag} (final.json exists)")
            continue
        cmd = [sys.executable, os.path.join(HERE, "train.py")]
        for k, v in job.items():
            cmd += [f"--{k}", str(v)]
        say(f"[{i + 1}/{len(jobs)}] RUN {tag}: {' '.join(cmd[2:])}")
        t0 = time.time()
        r = subprocess.run(cmd)
        dt = (time.time() - t0) / 60
        say(
            f"[{i + 1}/{len(jobs)}] {'OK' if r.returncode == 0 else 'FAIL(%d)' % r.returncode} {tag} in {dt:.1f} min"
        )
    say("queue complete")


if __name__ == "__main__":
    main()
