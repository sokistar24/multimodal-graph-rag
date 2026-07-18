"""
run_matrix.py — schedules the full 4-model x 3-question-type ablation.

Runs sequentially (one CLIP load at a time, no GPU contention, no interleaved
output). Skips runs that already produced a summary CSV, so it is safe to
Ctrl-C and restart. Every run's stdout is teed to logs/.

Usage:
    python run_matrix.py                 # run everything not yet done
    python run_matrix.py --dry-run       # print the plan, run nothing
    python run_matrix.py --force         # re-run even if summary exists
    python run_matrix.py --only gemini-flash-lite
    python run_matrix.py --skip llama4-maverick
"""
import argparse, os, subprocess, sys, time, datetime, glob

MODELS = ["gpt4o-mini", "gemini-flash-lite", "llama4-scout", "llama4-maverick"]

QSETS = [
    "questions_publaynet_text.json",
    "questions_publaynet_multihop.json",
    "questions_publaynet_figures.json",
]

RESULTS = "results"
LOGS    = "logs"


def qname(qfile):
    return os.path.splitext(os.path.basename(qfile))[0]


def already_done(model, qfile):
    """A run is done if a summary CSV exists for that (qset, model) pair."""
    pat = os.path.join(RESULTS, f"summary_{qname(qfile)}_{model}_*.csv")
    return sorted(glob.glob(pat))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if a summary CSV already exists")
    ap.add_argument("--only", nargs="+", default=None,
                    help="only these models")
    ap.add_argument("--skip", nargs="+", default=[],
                    help="skip these models")
    args = ap.parse_args()

    models = args.only if args.only else MODELS
    models = [m for m in models if m not in args.skip]

    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    # ---- build the plan ----
    plan, skipped = [], []
    for model in models:
        for qfile in QSETS:
            if not os.path.exists(qfile):
                print(f"!! missing question file, skipping: {qfile}")
                continue
            done = already_done(model, qfile)
            if done and not args.force:
                skipped.append((model, qfile, done[-1]))
            else:
                plan.append((model, qfile))

    print("=" * 72)
    print(f"MATRIX: {len(models)} models x {len(QSETS)} question sets")
    print("=" * 72)
    if skipped:
        print(f"\nALREADY DONE ({len(skipped)}) — skipping (use --force to redo):")
        for m, q, csv in skipped:
            print(f"  {m:<20} {qname(q):<32} {os.path.basename(csv)}")
    print(f"\nTO RUN ({len(plan)}):")
    for m, q in plan:
        print(f"  {m:<20} {qname(q)}")
    print()

    if args.dry_run:
        print("dry run — nothing executed.")
        return

    if not plan:
        print("nothing to do.")
        return

    # ---- execute ----
    t_start = time.time()
    failures = []
    for i, (model, qfile) in enumerate(plan, 1):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log = os.path.join(LOGS, f"{qname(qfile)}_{model}_{stamp}.log")
        # -u forces the child unbuffered. Without it Python block-buffers when
        # stdout is a pipe, so compare_all's progress ("12/35 ...") accumulates
        # in an 8KB buffer and only appears when the run finishes — the run looks
        # hung for minutes at a time.
        cmd = [sys.executable, "-u", "compare_all.py", qfile, "--model", model]

        print("=" * 72)
        print(f"[{i}/{len(plan)}]  {model}  |  {qname(qfile)}")
        print(f"  cmd : {' '.join(cmd)}")
        print(f"  log : {log}")
        print("=" * 72)

        t0 = time.time()
        with open(log, "w", encoding="utf-8") as fh:
            fh.write(f"# {' '.join(cmd)}\n# started {stamp}\n\n")
            fh.flush()
            # Force UTF-8 on the child's stdout.
            #
            # Windows consoles default to cp1252. PubLayNet is biomedical text,
            # so question previews contain Greek letters (TGF-beta prints as a
            # real beta char). compare_all's progress line then dies with
            # UnicodeEncodeError — AFTER every API call has been made and paid
            # for, but BEFORE the summary CSV is written. A 12-minute run and
            # its API spend, lost to a print statement.
            #
            # errors="replace" below only covers OUR read of the pipe; it does
            # nothing for the child's own encode. This env var is what fixes it.
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=0,
                env=env,
            )
            # Read by CHARACTER, not by line. compare_all's progress counter
            # redraws with \r and never emits \n until the run ends, so
            # `for line in proc.stdout` would block until then and print the
            # whole run in one dump at the end.
            buf = []
            while True:
                ch = proc.stdout.read(1)
                if not ch:
                    break
                sys.stdout.write(ch)
                sys.stdout.flush()          # \r progress needs an explicit flush
                buf.append(ch)
                if ch in "\n\r":
                    fh.write("".join(buf))  # log in line-ish batches
                    fh.flush()
                    buf = []
            if buf:
                fh.write("".join(buf))
                fh.flush()
            proc.wait()

        dt = time.time() - t0
        if proc.returncode != 0:
            print(f"\n!! FAILED (exit {proc.returncode}) after {dt:.0f}s — see {log}")
            print("!! continuing with the rest of the matrix.\n")
            failures.append((model, qfile, proc.returncode, log))
        else:
            print(f"\n-- done in {dt/60:.1f} min\n")

    # ---- report ----
    total = (time.time() - t_start) / 60
    print("=" * 72)
    print(f"MATRIX COMPLETE — {len(plan)-len(failures)}/{len(plan)} succeeded, "
          f"{total:.0f} min total")
    print("=" * 72)
    if failures:
        print("\nFAILED RUNS:")
        for m, q, rc, log in failures:
            print(f"  {m:<20} {qname(q):<32} exit {rc}   {log}")
        print("\nRe-run just these with:  python run_matrix.py --only <model>")
    else:
        print("\nall clean.")

    print(f"\nSummaries in {RESULTS}\\ — collect with:")
    print(f"  python collect_results.py")


if __name__ == "__main__":
    main()
