"""Checkpoint restore that reads a rank's DCP items on several threads and commits them on the calling thread.

Torch's FileSystemReader reads a rank's files one after another on one stream: about 120 MB/s per rank from the
shared volume. Here worker threads each read a run of the rank's items on a stream of their own, as the stock reader
does, and the calling thread hands each item to the planner exactly as the stock reader does, one at a time, with
resolve and commit adjacent (Megatron's planner keeps state between them). A bounded queue caps the items read but
not yet committed, so buffered memory stays near threads x the largest item.
"""

import io
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import torch
from torch.distributed.checkpoint import FileSystemReader, filesystem

logger = logging.getLogger(__name__)


def runs(reader, plan, threads):
    """The plan's items by file in offset order, each file cut into up to `threads` runs of about equal bytes."""
    per_file = {}
    for request in plan.items:
        per_file.setdefault(reader.storage_data[request.storage_index].relative_path, []).append(request)
    units = []
    for relative_path, requests in per_file.items():
        requests.sort(key=lambda request: reader.storage_data[request.storage_index].offset)
        share = sum(reader.storage_data[request.storage_index].length for request in requests) / threads
        run, filled = [], 0
        for request in requests:
            run.append(request)
            filled += reader.storage_data[request.storage_index].length
            if filled >= share:
                units.append((relative_path, run))
                run, filled = [], 0
        if run:
            units.append((relative_path, run))
    return units


def read_run(reader, relative_path, requests, results, stop):
    """Read one run of items on its own stream, putting (request, value, bytes) on `results` until stopped."""
    with reader.fs.create_stream(reader.fs.concat_path(reader.path, relative_path), "rb") as stream:
        for request in requests:
            item = reader.storage_data[request.storage_index]
            view = reader.transforms.transform_load_stream(
                request, item.transform_descriptors or (), reader._slice_file(stream, item)
            )
            if request.type == filesystem.LoadItemType.BYTE_IO:
                value = io.BytesIO(view.read(-1))
            else:
                seekable = view if view.seekable() else io.BytesIO(view.read(-1))
                tensor = torch.load(seekable, map_location="cpu", weights_only=True)
                value = filesystem.narrow_tensor_by_index(tensor, request.storage_offsets, request.lengths)
            while True:
                if stop.is_set():
                    return
                try:
                    results.put((request, value, item.length), timeout=1)
                    break
                except queue.Full:
                    continue


def commit(planner, request, value):
    """Hand one item to the planner as the stock reader does: bytes loaded, or a tensor resolved, copied, committed."""
    if request.type == filesystem.LoadItemType.BYTE_IO:
        planner.load_bytes(request, value)
        return
    target = planner.resolve_tensor(request).detach()
    if target.size() != value.size():
        raise AssertionError(f"req {request.storage_index} mismatch sizes {target.size()} vs {value.size()}")
    target.copy_(value)
    planner.commit_tensor(request, target)


def read_data(reader, plan, planner, threads):
    """FileSystemReader.read_data on `threads` threads: every item read once, committed on the calling thread."""
    started, remaining, read = time.monotonic(), len(plan.items), 0
    results, stop = queue.Queue(maxsize=2 * threads), threading.Event()
    pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="checkpoint-read")
    try:
        futures = [pool.submit(read_run, reader, path, run, results, stop) for path, run in runs(reader, plan, threads)]
        while remaining:
            try:
                request, value, length = results.get(timeout=1)
            except queue.Empty:
                for future in futures:
                    if future.done() and future.exception() is not None:
                        raise future.exception() from None
                continue
            commit(planner, request, value)
            remaining, read = remaining - 1, read + length
    finally:
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    seconds = time.monotonic() - started
    logger.info("Checkpoint read on rank %d: %d bytes in %.1f s on %d threads", rank, read, seconds, threads)
    done = torch.futures.Future()
    done.set_result(None)
    return done


@contextmanager
def threaded_reads(threads):
    """Within the block every FileSystemReader, Megatron's cached-metadata subclass included, reads on `threads`
    threads; one thread keeps torch's own reader."""
    if threads is None or threads <= 1:
        yield
        return
    stock = FileSystemReader.read_data
    FileSystemReader.read_data = lambda reader, plan, planner: read_data(reader, plan, planner, threads)
    try:
        yield
    finally:
        FileSystemReader.read_data = stock
