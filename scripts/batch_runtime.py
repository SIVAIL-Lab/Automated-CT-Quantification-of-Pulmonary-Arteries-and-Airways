"""Signal handling and process cleanup for isolated batch workers."""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import signal
import threading
import time


class BatchInterrupted(KeyboardInterrupt):
    def __init__(self, signum: int):
        super().__init__(f"Batch interrupted by signal {signum}")
        self.signum = signum


@contextmanager
def _interrupt_handlers():
    previous = {}
    if threading.current_thread() is threading.main_thread():
        def interrupt(signum, frame):
            # Cleanup must survive a second Ctrl-C or supervisor TERM.
            for item in (signal.SIGINT, signal.SIGTERM):
                signal.signal(item, signal.SIG_IGN)
            raise BatchInterrupted(signum)
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, interrupt)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


@contextmanager
def _child_subreaper():
    """Let this Linux driver reap workers' orphaned inference children."""
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot read child-subreaper setting")
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot enable child subreaper")
    try:
        yield
    finally:
        libc.prctl(36, previous.value, 0, 0, 0)


def _descendant_pids() -> set[int]:
    """Read the driver's descendants, including adopted orphan children."""
    children = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            children.setdefault(int(fields[1]), []).append(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    found, pending = set(), [os.getpid()]
    while pending:
        for child in children.get(pending.pop(), []):
            if child not in found:
                found.add(child)
                pending.append(child)
    return found


def _reap_pid(pid: int) -> None:
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass


def _terminate_serial_descendants(previous: set[int]) -> None:
    # Children may share the caller's group; target only owned descendants.
    owned = _descendant_pids() - previous
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in owned:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 2.0
        while owned and time.monotonic() < deadline:
            for pid in owned:
                _reap_pid(pid)
            owned = _descendant_pids() - previous
            if owned:
                time.sleep(.05)
    if owned:
        raise RuntimeError(f"Batch descendants did not terminate: {sorted(owned)}")


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def _terminate_workers(jobs: list[dict], *, grace: float = 3.0) -> None:
    """Terminate and reap entire groups, even after their worker leader died."""
    if not jobs:
        return
    for sig, duration in ((signal.SIGTERM, grace), (signal.SIGKILL, 3.0)):
        for job in jobs:
            try:
                os.killpg(job["process"].pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            remaining = False
            for job in jobs:
                process = job["process"]
                if process.poll() is not None:
                    try:
                        while os.waitpid(-process.pid, os.WNOHANG)[0]:
                            pass
                    except ChildProcessError:
                        pass
                remaining |= _group_exists(process.pid)
            if not remaining:
                break
            time.sleep(.05)
        if not remaining:
            break
    for job in jobs:
        job["process"].wait(timeout=1)
        job["log_handle"].close()
    if any(_group_exists(job["process"].pid) for job in jobs):
        raise RuntimeError("A case process group did not terminate")
