"""End-to-end loopback transfer benchmarks at increasing concurrency.

Uses pytest-benchmark for timing/statistics and concurrent.futures for the
worker threads, rather than hand-rolled timing loops and manual thread
bookkeeping. That removes three defects the previous script had: a per-thread
`join(timeout=...)` that made the effective deadline `concurrency * timeout`,
only the first of several worker exceptions being surfaced, and receivers
leaking when a worker overran its deadline.

Not collected by a normal `pytest` run (`testpaths` is `tests`):

    uv run --group bench pytest benchmarks/test_concurrent_transfers.py
"""

from __future__ import annotations

import contextlib
import hashlib
import socket
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import pytest

from ssync.space_sync.receiver import SpaceSyncReceiver
from ssync.space_sync.sender import SpaceSyncSender
from ssync.space_sync.types import ReceiverConfig, SenderConfig

CONCURRENCY_LEVELS = [1, 2, 4]
FILE_SIZE_BYTES = 8 * 1024 * 1024
CHUNK_SIZE = 4096
# A per-sender cap keeps the benchmark measuring transport work rather than how
# fast an unpaced sender can overrun the receiver's fixed-size kernel socket
# buffer, which is a kernel-tuning artifact rather than a property of ssync.
MAX_DATA_RATE_BPS = 100_000_000
TRIAL_TIMEOUT_S = 120.0


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _payload(size_bytes: int, seed: int) -> bytes:
    block = hashlib.sha256(f"ssync-benchmark-{seed}".encode()).digest() * 32_768
    return (block * (size_bytes // len(block) + 1))[:size_bytes]


@contextlib.contextmanager
def _receiver(output_dir: Path, port: int) -> Iterator[SpaceSyncReceiver]:
    """Guarantees the receiver's threads and socket are released."""
    receiver = SpaceSyncReceiver(
        bind_host="127.0.0.1",
        bind_port=port,
        config=ReceiverConfig(output_dir=output_dir, enable_feedback=True),
    )
    receiver.start()
    try:
        yield receiver
    finally:
        receiver.stop()


def _transfer_one(source: Path, port: int) -> bool:
    sender = SpaceSyncSender(
        config=SenderConfig(
            chunk_size=CHUNK_SIZE,
            enable_feedback=True,
            max_data_rate_bps=MAX_DATA_RATE_BPS,
        )
    )
    return sender.send_file(source, "127.0.0.1", port).completed


@pytest.mark.parametrize("concurrency", CONCURRENCY_LEVELS)
def test_concurrent_transfers(benchmark, tmp_path: Path, concurrency: int) -> None:
    """Aggregate throughput of N simultaneous transfers in one process."""
    sources = []
    for index in range(concurrency):
        source = tmp_path / f"src-{index}.bin"
        source.write_bytes(_payload(FILE_SIZE_BYTES, index))
        sources.append(source)

    # Ports are allocated once, here in the main thread. The previous script
    # had each worker thread call the free-port helper concurrently, which let
    # two receivers race onto the same kernel-assigned port.
    ports = [_free_udp_port() for _ in range(concurrency)]

    def run_trial() -> None:
        with contextlib.ExitStack() as stack:
            for index, port in enumerate(ports):
                output_dir = tmp_path / f"rx-{index}"
                output_dir.mkdir(exist_ok=True)
                stack.enter_context(_receiver(output_dir, port))
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [
                    pool.submit(_transfer_one, sources[i], ports[i])
                    for i in range(concurrency)
                ]
                # One deadline for the whole trial, and every worker's
                # exception is surfaced rather than just the first.
                done, pending = wait(futures, timeout=TRIAL_TIMEOUT_S)
                assert not pending, f"{len(pending)} transfer(s) exceeded the deadline"
                for future in done:
                    assert future.result() is True, "transfer did not complete"

    benchmark.extra_info["gil_enabled"] = getattr(
        sys, "_is_gil_enabled", lambda: True
    )()
    benchmark.extra_info["python"] = sys.version.split()[0]
    benchmark.extra_info["concurrency"] = concurrency
    benchmark.extra_info["file_size_bytes"] = FILE_SIZE_BYTES
    benchmark.pedantic(run_trial, rounds=3, warmup_rounds=0)
    mean = benchmark.stats.stats.mean
    if mean > 0:
        benchmark.extra_info["aggregate_bits_per_sec"] = (
            concurrency * FILE_SIZE_BYTES * 8
        ) / mean
