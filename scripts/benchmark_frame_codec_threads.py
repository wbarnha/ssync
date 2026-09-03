#!/usr/bin/env python3
"""Pure-CPU micro-benchmark: encode/decode data-chunk frames on N threads.

This isolates raw interpreter threading behavior from ssync's networking
code (no sockets, no kernel UDP buffers, no repair protocol timing) so it
directly answers: does splitting CPU-bound, pure-Python frame codec work
(the same `encode_data_chunk`/`decode_frame`/`decode_data_chunk` calls the
sender and receiver make per chunk) across threads actually scale with core
count under a free-threaded build, versus a standard build where the GIL
serializes Python bytecode execution across threads?

Examples:
  uv run --python 3.13.7 python scripts/benchmark_frame_codec_threads.py
  uv run --python 3.13.7+freethreaded python scripts/benchmark_frame_codec_threads.py
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ssync.space_sync.frames import decode_data_chunk, decode_frame, encode_data_chunk


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--thread-counts",
        default="1,2,4",
        help="Comma-separated thread counts to test",
    )
    parser.add_argument(
        "--ops-per-thread",
        type=int,
        default=200_000,
        help="Encode+decode round trips per thread",
    )
    parser.add_argument("--payload-bytes", type=int, default=1024)
    return parser.parse_args()


def _worker(ops: int, payload: bytes, transfer_id: bytes) -> None:
    for i in range(ops):
        frame = encode_data_chunk(transfer_id, i & 0xFFFFFFFF, payload)
        parsed = decode_frame(frame)
        decode_data_chunk(parsed.payload)


def _run(thread_count: int, ops_per_thread: int, payload_bytes: int) -> float:
    payload = os.urandom(payload_bytes)
    transfer_id = os.urandom(16)
    threads = [
        threading.Thread(target=_worker, args=(ops_per_thread, payload, transfer_id))
        for _ in range(thread_count)
    ]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return time.perf_counter() - started


def main() -> int:
    args = _parse_args()
    thread_counts = [int(part.strip()) for part in args.thread_counts.split(",") if part.strip()]

    baseline_s: float | None = None
    for thread_count in thread_counts:
        elapsed_s = _run(thread_count, args.ops_per_thread, args.payload_bytes)
        total_ops = thread_count * args.ops_per_thread
        ops_per_s = total_ops / elapsed_s if elapsed_s > 0 else 0.0
        if baseline_s is None:
            baseline_s = elapsed_s
        speedup = baseline_s / elapsed_s if elapsed_s > 0 else 0.0
        print(
            f"threads={thread_count} elapsed_s={elapsed_s:.3f} "
            f"ops_per_s={ops_per_s:,.0f} speedup_vs_1_thread={speedup:.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
