#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
# ruff: noqa: I001
"""Benchmark N concurrent feedback-mode loopback transfers in one process.

This exists to compare CPython's standard (GIL) build against a
free-threaded (PEP 703, no-GIL) build: a single sender/receiver pair is
mostly bound by one core's worth of Python bytecode (frame construction and
`socket.sendto`/`recvfrom` calls), so running several pairs concurrently in
threads shows whether aggregate throughput scales with core count or
plateaus at the GIL's single-core ceiling.

Feedback (repair) mode is used, with a per-sender rate cap, rather than an
uncapped open loop: an uncapped sender can outrun the receiver's fixed-size
kernel UDP socket buffer once real thread parallelism is available, which
drops datagrams for kernel-buffering reasons unrelated to the interpreter
being measured. Repair mode absorbs that transient loss instead of
corrupting the result.

Examples:
  uv run --python 3.13.7 python scripts/benchmark_concurrent_transfers.py \\
      --concurrency 1,2,4 --output-json
  uv run --python 3.13.7+freethreaded python scripts/benchmark_concurrent_transfers.py \\
      --concurrency 1,2,4 --output-json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ssync.space_sync.receiver import SpaceSyncReceiver
from ssync.space_sync.sender import SpaceSyncSender
from ssync.space_sync.types import DEFAULT_CHUNK_SIZE, ReceiverConfig, SendResult, SenderConfig


@dataclass(slots=True)
class TransferOutcome:
    index: int
    success: bool
    duration_s: float
    achieved_bps: float
    hash_match: bool
    send_completed: bool


@dataclass(slots=True)
class ConcurrencyResult:
    concurrency: int
    trial_index: int
    wall_time_s: float
    aggregate_bps: float
    all_success: bool
    per_transfer: list[TransferOutcome]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run N concurrent open-loop loopback transfers and report "
            "aggregate throughput scaling versus concurrency"
        )
    )
    parser.add_argument(
        "--concurrency",
        default="1,2,4",
        help="Comma-separated list of concurrent transfer counts to test",
    )
    parser.add_argument("--trials", type=int, default=3, help="Trials per concurrency level")
    parser.add_argument("--file-size-mib", type=int, default=32, help="Per-transfer payload size")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--max-data-rate-bps",
        type=int,
        default=100_000_000,
        help=(
            "Per-sender rate cap in bits/sec; 0 means unlimited. A cap is used "
            "by default because uncapped open-loop UDP can overrun the "
            "receiver's fixed-size kernel socket buffer under real parallelism "
            "and lose datagrams, which is a kernel-buffering artifact rather "
            "than an interpreter performance difference"
        ),
    )
    parser.add_argument(
        "--socket-rcvbuf-bytes",
        type=int,
        default=8 * 1024 * 1024,
        help="Receiver UDP socket receive-buffer size",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=60.0,
        help="Max seconds to wait for all transfers in a trial to finish",
    )
    parser.add_argument("--output-json", action="store_true")
    return parser.parse_args()


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_payload(path: Path, size_bytes: int, seed: int) -> None:
    block = hashlib.sha256(f"ssync-concurrency-benchmark-{seed}".encode()).digest() * 32_768
    with path.open("wb") as stream:
        remaining = size_bytes
        while remaining > 0:
            piece = block[: min(len(block), remaining)]
            stream.write(piece)
            remaining -= len(piece)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _run_one_transfer(
    *,
    index: int,
    work_dir: Path,
    file_size_bytes: int,
    chunk_size: int,
    max_data_rate_bps: int,
    socket_rcvbuf_bytes: int,
    outcomes: list[TransferOutcome | None],
    errors: dict[int, BaseException],
) -> None:
    try:
        source = work_dir / f"src-{index}.bin"
        _write_payload(source, file_size_bytes, seed=index)
        source_hash = _sha256(source)

        receiver_dir = work_dir / f"rx-{index}"
        receiver_dir.mkdir(parents=True, exist_ok=True)
        port = _free_udp_port()
        receiver = SpaceSyncReceiver(
            bind_host="127.0.0.1",
            bind_port=port,
            config=ReceiverConfig(
                output_dir=receiver_dir,
                enable_feedback=True,
                socket_rcvbuf_bytes=socket_rcvbuf_bytes,
            ),
        )
        receiver.start()
        time.sleep(0.05)
        sender = SpaceSyncSender(
            config=SenderConfig(
                chunk_size=chunk_size,
                enable_feedback=True,
                max_data_rate_bps=max_data_rate_bps,
            )
        )
        started = time.monotonic()
        try:
            send_result: SendResult = sender.send_file(source, "127.0.0.1", port)
            duration_s = time.monotonic() - started
            target = receiver_dir / source.name
            deadline = time.monotonic() + 10.0
            while not target.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            hash_match = target.exists() and _sha256(target) == source_hash
            success = bool(send_result.completed and hash_match)
            achieved_bps = (file_size_bytes * 8) / duration_s if duration_s > 0 else 0.0
            outcomes[index] = TransferOutcome(
                index=index,
                success=success,
                duration_s=duration_s,
                achieved_bps=achieved_bps,
                hash_match=hash_match,
                send_completed=send_result.completed,
            )
        finally:
            receiver.stop()
    except BaseException as exc:  # pragma: no cover - surfaces back to caller
        errors[index] = exc


def _run_trial(
    *,
    concurrency: int,
    trial_index: int,
    work_root: Path,
    args: argparse.Namespace,
) -> ConcurrencyResult:
    file_size_bytes = args.file_size_mib * 1024 * 1024
    work_dir = work_root / f"c{concurrency}-t{trial_index}"
    work_dir.mkdir(parents=True, exist_ok=True)

    outcomes: list[TransferOutcome | None] = [None] * concurrency
    errors: dict[int, BaseException] = {}
    threads = [
        threading.Thread(
            target=_run_one_transfer,
            kwargs={
                "index": i,
                "work_dir": work_dir,
                "file_size_bytes": file_size_bytes,
                "chunk_size": args.chunk_size,
                "max_data_rate_bps": args.max_data_rate_bps,
                "socket_rcvbuf_bytes": args.socket_rcvbuf_bytes,
                "outcomes": outcomes,
                "errors": errors,
            },
            daemon=True,
        )
        for i in range(concurrency)
    ]

    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=args.timeout_s)
    wall_time_s = time.monotonic() - started

    if errors:
        first_index = next(iter(errors))
        raise errors[first_index]

    resolved: list[TransferOutcome] = []
    for outcome in outcomes:
        if outcome is None:
            raise RuntimeError("a transfer thread did not finish within the timeout")
        resolved.append(outcome)

    total_bytes = concurrency * file_size_bytes
    aggregate_bps = (total_bytes * 8) / wall_time_s if wall_time_s > 0 else 0.0
    return ConcurrencyResult(
        concurrency=concurrency,
        trial_index=trial_index,
        wall_time_s=wall_time_s,
        aggregate_bps=aggregate_bps,
        all_success=all(o.success for o in resolved),
        per_transfer=resolved,
    )


def _format_gbps(bps: float) -> str:
    return f"{bps / 1_000_000_000:.3f} Gbps"


def main() -> int:
    args = _parse_args()
    concurrency_levels = [int(part.strip()) for part in args.concurrency.split(",") if part.strip()]

    all_results: list[ConcurrencyResult] = []
    with tempfile.TemporaryDirectory(prefix="ssync-concurrency-bench-") as tmp:
        work_root = Path(tmp)
        for concurrency in concurrency_levels:
            trial_results = []
            for trial_index in range(args.trials):
                result = _run_trial(
                    concurrency=concurrency,
                    trial_index=trial_index,
                    work_root=work_root,
                    args=args,
                )
                trial_results.append(result)
                all_results.append(result)
                print(
                    f"concurrency={concurrency} trial={trial_index} "
                    f"wall_time_s={result.wall_time_s:.3f} "
                    f"aggregate={_format_gbps(result.aggregate_bps)} "
                    f"all_success={result.all_success}"
                )
            best = max(trial_results, key=lambda r: r.aggregate_bps)
            print(
                f"-> concurrency={concurrency} best_aggregate={_format_gbps(best.aggregate_bps)} "
                f"best_wall_time_s={best.wall_time_s:.3f}"
            )

    if args.output_json:
        print(json.dumps([asdict(result) for result in all_results], indent=2))

    return 0 if all(result.all_success for result in all_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
