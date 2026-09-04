"""Frame-codec microbenchmarks, including free-threaded (PEP 703) scaling.

These are pytest-benchmark benchmarks rather than a hand-rolled timing loop:
the library handles warmup, repeated rounds, and reports median/StdDev/IQR and
outlier counts, which matters here because single-run timings on this workload
are noisy enough to invert the conclusion.

Not collected by a normal `pytest` run - `testpaths` is `tests`, so these only
run when pointed at explicitly:

    uv run --group bench pytest benchmarks/ --benchmark-columns=median,stddev,ops

Compare interpreters by saving and diffing runs:

    uv run --group bench --python 3.14 pytest benchmarks/ --benchmark-save=gil
    uv run --group bench --python 3.14t pytest benchmarks/ --benchmark-save=ft
    uv run --group bench pytest-benchmark compare gil ft
"""

from __future__ import annotations

import struct
import sys
import threading
from collections.abc import Callable

import pytest

from ssync.space_sync.frames import (
    DATA_FIXED_STRUCT,
    HEADER_STRUCT,
    decode_data_chunk,
    decode_frame,
    encode_data_chunk,
)

OPS_PER_THREAD = 50_000
PAYLOAD_BYTES = 1024
THREAD_COUNTS = [1, 2, 4]
HEADER_ARGS = (b"SS", 1, 2, 0, 0, 1024)


def _gil_enabled() -> bool:
    return getattr(sys, "_is_gil_enabled", lambda: True)()


def _run_on_threads(work: Callable[[int], None], threads: int, ops: int) -> None:
    workers = [threading.Thread(target=work, args=(ops,)) for _ in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()


def _record(benchmark, threads: int, ops_per_thread: int) -> None:
    """Attach aggregate throughput, which is what thread scaling is read from."""
    benchmark.extra_info["gil_enabled"] = _gil_enabled()
    benchmark.extra_info["python"] = sys.version.split()[0]
    benchmark.extra_info["threads"] = threads
    benchmark.extra_info["ops_per_thread"] = ops_per_thread
    mean = benchmark.stats.stats.mean
    if mean > 0:
        benchmark.extra_info["aggregate_ops_per_sec"] = (threads * ops_per_thread) / mean


# --- the codec itself --------------------------------------------------------


def _roundtrip(ops: int) -> None:
    # Buffers are created per thread, mirroring a sender that reads a fresh
    # chunk per frame rather than re-sending one shared buffer.
    payload = bytes(PAYLOAD_BYTES)
    transfer_id = bytes(16)
    for index in range(ops):
        parsed = decode_frame(encode_data_chunk(transfer_id, index & 0xFFFFFFFF, payload))
        decode_data_chunk(parsed.payload)


def _encode_only(ops: int) -> None:
    payload = bytes(PAYLOAD_BYTES)
    transfer_id = bytes(16)
    for index in range(ops):
        encode_data_chunk(transfer_id, index & 0xFFFFFFFF, payload)


def _decode_only(ops: int) -> None:
    frame = encode_data_chunk(bytes(16), 1, bytes(PAYLOAD_BYTES))
    for _ in range(ops):
        decode_frame(frame)


@pytest.mark.parametrize("threads", THREAD_COUNTS)
@pytest.mark.parametrize(
    ("name", "work"),
    [("roundtrip", _roundtrip), ("encode", _encode_only), ("decode", _decode_only)],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_codec(benchmark, name: str, work: Callable[[int], None], threads: int) -> None:
    """Codec throughput at 1/2/4 threads; compare rows to read thread scaling."""
    benchmark.group = f"codec-{name}"
    benchmark.pedantic(
        _run_on_threads,
        args=(work, threads, OPS_PER_THREAD),
        rounds=5,
        warmup_rounds=1,
    )
    _record(benchmark, threads, OPS_PER_THREAD)


# --- why the codec does not scale linearly -----------------------------------
#
# struct.pack's scaling is governed by whether the Struct object is shared
# between threads: a shared object is refcounted by non-owner threads on every
# call, which turns into a contended atomic on one cache line. Keep both
# variants as runnable benchmarks so the claim stays checkable.


def _pack_shared_struct(ops: int) -> None:
    packer = HEADER_STRUCT  # module-level singleton, shared by every thread
    for _ in range(ops):
        packer.pack(*HEADER_ARGS)


def _pack_per_thread_struct(ops: int) -> None:
    packer = struct.Struct(HEADER_STRUCT.format)  # private to this thread
    for _ in range(ops):
        packer.pack(*HEADER_ARGS)


@pytest.mark.parametrize("threads", THREAD_COUNTS)
@pytest.mark.parametrize(
    ("name", "work"),
    [("shared", _pack_shared_struct), ("per_thread", _pack_per_thread_struct)],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_struct_pack_sharing(
    benchmark, name: str, work: Callable[[int], None], threads: int
) -> None:
    """Shared vs per-thread Struct: the dominant limit on codec thread scaling."""
    benchmark.group = "struct-pack-sharing"
    benchmark.pedantic(
        _run_on_threads,
        args=(work, threads, OPS_PER_THREAD),
        rounds=5,
        warmup_rounds=1,
    )
    _record(benchmark, threads, OPS_PER_THREAD)
    benchmark.extra_info["struct_sharing"] = name


SHARED_BUFFER = HEADER_STRUCT.pack(*HEADER_ARGS)


def _unpack_shared(ops: int) -> None:
    # Both the Struct and the buffer are shared by every thread.
    for _ in range(ops):
        HEADER_STRUCT.unpack_from(SHARED_BUFFER, 0)


def _unpack_per_thread(ops: int) -> None:
    # Both are private to this thread.
    unpacker = struct.Struct(HEADER_STRUCT.format)
    buffer = unpacker.pack(*HEADER_ARGS)
    for _ in range(ops):
        unpacker.unpack_from(buffer, 0)


@pytest.mark.parametrize("threads", THREAD_COUNTS)
@pytest.mark.parametrize(
    ("name", "work"),
    [("shared", _unpack_shared), ("per_thread", _unpack_per_thread)],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_struct_unpack_sharing(
    benchmark, name: str, work: Callable[[int], None], threads: int
) -> None:
    """Same contrast for unpack: everything shared vs everything thread-private."""
    benchmark.group = "struct-unpack-sharing"
    benchmark.pedantic(
        _run_on_threads,
        args=(work, threads, OPS_PER_THREAD),
        rounds=5,
        warmup_rounds=1,
    )
    _record(benchmark, threads, OPS_PER_THREAD)
    benchmark.extra_info["sharing"] = name


def test_data_fixed_struct_is_shared() -> None:
    """Guard the premise above: these really are module-level singletons."""
    assert encode_data_chunk.__globals__["DATA_FIXED_STRUCT"] is DATA_FIXED_STRUCT
    assert encode_data_chunk.__globals__["HEADER_STRUCT"] is HEADER_STRUCT
