"""Benchmark for memcached prediction read/write performance.

Requires a running memcached instance (default: localhost:11211).
Usage:
    maturin develop --release
    python benchmarks/bench_post_preds.py [--memcached-host HOST] [--zmq-url URL]
"""

import argparse
import time

import numpy as np

from alpha_cc.engine import Board, NNPred, PredDBChannel

try:
    from alpha_cc.engine import post_preds_from_logits

    HAS_POST_PREDS_FROM_LOGITS = True
except ImportError:
    HAS_POST_PREDS_FROM_LOGITS = False

BATCH_SIZES = [1, 10, 50, 100, 256, 512]
N_WARMUP = 5
N_REPEATS = 30
BOARD_SIZE = 5


def make_test_data(batch_size: int, board_size: int):
    boards = [Board(board_size) for _ in range(batch_size)]
    nn_preds = [
        NNPred(
            [1.0 / len(b.get_moves())] * len(b.get_moves()),
            0.0,
        )
        for b in boards
    ]
    s = board_size * 2 - 1
    logits_flat = np.random.randn(batch_size, s, s, s, s).astype(np.float32).ravel().tolist()
    values_flat = np.random.randn(batch_size).astype(np.float32).tolist()
    return boards, nn_preds, logits_flat, values_flat, s


def bench(fn, n_warmup, n_repeats):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def report(label, times):
    arr = np.array(times) * 1000  # ms
    print(
        f"  {label:45s}"
        f"  avg={arr.mean():8.3f}ms"
        f"  std={arr.std():7.3f}ms"
        f"  p50={np.median(arr):8.3f}ms"
        f"  p99={np.percentile(arr, 99):8.3f}ms"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memcached-host", default="localhost")
    parser.add_argument("--zmq-url", default="localhost")
    args = parser.parse_args()

    pred_db = PredDBChannel(args.zmq_url, args.memcached_host, 0)
    pred_db.flush_preds()

    header = f"{'':2s}{'method':45s}  {'avg':>10s}  {'std':>9s}  {'p50':>10s}  {'p99':>10s}"

    # --- WRITE benchmarks ---
    print(f"\n=== WRITE benchmarks (board_size={BOARD_SIZE}, warmup={N_WARMUP}, repeats={N_REPEATS}) ===\n")
    print(header)
    print("-" * 100)

    for batch_size in BATCH_SIZES:
        boards, nn_preds, logits_flat, values_flat, s = make_test_data(batch_size, BOARD_SIZE)
        print(f"\nbatch_size={batch_size}")

        # post_preds (batch write via PredDBChannel)
        report(
            "post_preds",
            bench(lambda: pred_db.post_preds(boards, nn_preds), N_WARMUP, N_REPEATS),
        )

        # post_pred one-by-one (sequential baseline)
        def post_one_by_one():
            for b, p in zip(boards, nn_preds):
                pred_db.post_pred(b, p)

        report(
            "post_pred (one-by-one)",
            bench(post_one_by_one, N_WARMUP, N_REPEATS),
        )

        # post_preds_from_logits (Rust, if available)
        if HAS_POST_PREDS_FROM_LOGITS:
            report(
                "post_preds_from_logits",
                bench(
                    lambda: post_preds_from_logits(pred_db, logits_flat, values_flat, boards, s),
                    N_WARMUP,
                    N_REPEATS,
                ),
            )

    # --- READ benchmarks ---
    print(f"\n\n=== READ benchmarks (board_size={BOARD_SIZE}, warmup={N_WARMUP}, repeats={N_REPEATS}) ===\n")
    print(header)
    print("-" * 100)

    for batch_size in BATCH_SIZES:
        boards, nn_preds, _, _, _ = make_test_data(batch_size, BOARD_SIZE)
        # ensure predictions exist
        pred_db.post_preds(boards, nn_preds)
        print(f"\nbatch_size={batch_size}")

        # has_pred one-by-one
        def read_has_pred():
            for b in boards:
                pred_db.has_pred(b)

        report(
            "has_pred (one-by-one)",
            bench(read_has_pred, N_WARMUP, N_REPEATS),
        )

    pred_db.flush_preds()
    print("\nDone.")


if __name__ == "__main__":
    main()
