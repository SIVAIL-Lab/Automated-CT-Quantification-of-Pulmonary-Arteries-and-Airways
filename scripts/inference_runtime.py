#!/usr/bin/env python3
"""Run a single inference in a fresh process holding the shared GPU lock."""

from __future__ import annotations

import fcntl
from functools import wraps
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable


THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)
EXECUTION_POLICY_NAME = "isolated_deterministic_inference"
EXECUTION_POLICY_VERSION = 1
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


class InferencePolicyError(ValueError):
    """A strict-policy failure must bypass nnU-Net's RuntimeError CPU retry."""


def inference_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({name: "4" for name in THREAD_ENVIRONMENT})
    environment["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "4"
    # cuBLAS reads this when its workspace is initialized, before the API call.
    environment["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    return environment


def _torch_settings(torch) -> dict[str, Any]:
    return {
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_autocast_dtype": str(torch.get_autocast_dtype("cuda")),
        "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    }


def _require_preserved_precision(before: dict, after: dict) -> None:
    for name in ("cudnn_allow_tf32", "cuda_matmul_allow_tf32", "float32_matmul_precision",
                 "cuda_autocast_dtype", "allow_fp16_reduced_precision_reduction",
                 "allow_bf16_reduced_precision_reduction"):
        if after[name] != before[name]:
            raise RuntimeError(f"Inference changed the preserved precision setting {name}")


def _strict_torch_settings(torch) -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def _configure_deterministic_inference() -> tuple[Any, dict[str, Any]]:
    import torch

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError("Deterministic inference requires the cuBLAS workspace environment before torch import")
    if torch.cuda.is_initialized():
        raise RuntimeError("Deterministic inference policy must be installed before CUDA initialization")
    before = _torch_settings(torch)
    # nnU-Net reads its paths at import time; preserve TotalSegmentator's setup
    # order before importing the predictor and its runtime compatibility patches.
    from totalsegmentator.config import setup_nnunet

    setup_nnunet()
    from totalsegmentator.nnunet import nnUNetPredictor

    policy = {
        "name": EXECUTION_POLICY_NAME, "version": EXECUTION_POLICY_VERSION,
        "scope": "new isolated inference only", "settings_before": before,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "precision_policy": "Preserve nnU-Net mixed precision and existing TF32 settings",
        "precision_environment": {name: os.environ.get(name) for name in
                                  ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")},
        "random_seed_override": None, "predictor_initializations": [],
        "deterministic_error_cpu_retry": "disabled",
        "cpu_thread_environment": {name: os.environ.get(name) for name in
                                   (*THREAD_ENVIRONMENT, "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS")},
        "nr_thr_saving": 2, "nr_thr_resamp": 1,
    }
    original_init = nnUNetPredictor.__init__
    original_sliding_window = nnUNetPredictor._internal_predict_sliding_window_return_logits

    @wraps(original_sliding_window)
    def deterministic_sliding_window(self, *args, **kwargs):
        try:
            return original_sliding_window(self, *args, **kwargs)
        except RuntimeError as error:
            if "deterministic" in str(error).lower():
                raise InferencePolicyError(
                    f"Strict deterministic inference failed; CPU accumulation retry is disabled: {error}"
                ) from error
            raise

    @wraps(original_init)
    def deterministic_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        constructor_settings = _torch_settings(torch)
        # nnU-Net's CUDA constructor forces benchmark=True. Reapply after every
        # constructor, including the rough crop model, before any model forward.
        _strict_torch_settings(torch)
        effective = _torch_settings(torch)
        _require_preserved_precision(before, effective)
        policy["predictor_initializations"].append({
            "device": str(self.device), "constructor_cudnn_benchmark": constructor_settings["cudnn_benchmark"],
            "effective_settings": effective,
        })

    nnUNetPredictor.__init__ = deterministic_init
    nnUNetPredictor._internal_predict_sliding_window_return_logits = deterministic_sliding_window
    _strict_torch_settings(torch)
    policy["settings_configured"] = _torch_settings(torch)
    _require_preserved_precision(before, policy["settings_configured"])
    return torch, policy


def _record_execution_policy(path: Path, torch, policy: dict[str, Any]) -> None:
    after = _torch_settings(torch)
    _require_preserved_precision(policy["settings_before"], after)
    if (after["cudnn_benchmark"] or not after["cudnn_deterministic"]
            or not after["deterministic_algorithms"] or after["deterministic_warn_only"]):
        raise RuntimeError("Deterministic inference settings changed during the API call")
    if not policy["predictor_initializations"]:
        raise RuntimeError("TotalSegmentator did not use the guarded nnU-Net predictor")
    policy["settings_after"] = after
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("TotalSegmentator inference report must be a JSON object")
    report["execution_policy"] = policy
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".inference-policy-", delete=False,
                                     encoding="utf-8") as stream:
        temporary = Path(stream.name)
        try:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_isolated_inference(
    *, image_path: Path, output: Path, report: Path, device: str, quiet: bool,
    parameters: dict[str, Any], lock_path: Path, stage: str,
    stage_callback: Callable | None = None,
    max_concurrent_inference: int = 1,
) -> None:
    """Keep CUDA imports out of the caller; the child owns the lock until exit.

    The child deliberately stays in the caller's process group so the batch
    scheduler can terminate the complete case tree. The event pipe carries only
    progress/errors; inference stdout and stderr retain the case log streams.
    """
    _validate_inference_limit(max_concurrent_inference)
    payload = {
        "image_path": str(image_path.resolve()), "output": str(output.resolve()),
        "report": str(report.resolve()), "device": device, "quiet": quiet,
        "parameters": parameters, "lock_path": str(lock_path.resolve()),
        "max_concurrent_inference": max_concurrent_inference,
    }
    if stage_callback is not None:
        stage_callback(stage, "waiting_inference", None)
    reader, writer = os.pipe()
    child = None
    child_error = None
    try:
        child = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--child", str(writer)],
            stdin=subprocess.PIPE, pass_fds=(writer,), env=inference_environment(),
            text=True,
        )
        os.close(writer)
        writer = None
        child.stdin.write(json.dumps(payload, allow_nan=False))
        child.stdin.close()
        with os.fdopen(reader, encoding="utf-8") as events:
            reader = None
            for line in events:
                event = json.loads(line)
                if event["event"] == "error":
                    child_error = event["error"]
                elif stage_callback is not None:
                    stage_callback(stage, event["event"], event.get("details"))
        return_code = child.wait()
        if return_code:
            detail = f": {child_error}" if child_error else ""
            raise RuntimeError(f"Isolated TotalSegmentator exited with code {return_code}{detail}")
        if stage_callback is not None:
            stage_callback(stage, "inference_complete", {"pid": child.pid})
    except BaseException:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise
    finally:
        if reader is not None:
            os.close(reader)
        if writer is not None:
            os.close(writer)


def _event(descriptor: int, event: str, **fields: Any) -> None:
    try:
        os.write(descriptor, (json.dumps({"event": event, **fields}) + "\n").encode())
    except BrokenPipeError:
        # A killed case worker must not release this child's lock prematurely.
        pass


def _validate_inference_limit(value):
    if type(value) is not int or value not in (1, 2, 4):
        raise ValueError("max_concurrent_inference must be 1, 2 or 4")


def _acquire_pool_capacity(lock_path, capacity):
    """Lease one active pool capacity; conflicting live pools fail closed.

    A short exclusive guard serializes registry initialization. Every pooled
    child holds a shared lease through exit, including while waiting for a slot.
    When no leases remain, the next caller can atomically choose a new capacity.
    """
    guard = os.open(f"{lock_path}.capacity_guard", os.O_CREAT | os.O_RDWR, 0o600)
    lease = None
    try:
        fcntl.flock(guard, fcntl.LOCK_EX)
        lease = os.open(f"{lock_path}.capacity", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            registry = json.loads(os.pread(lease, 1024, 0).decode("utf-8"))
            if registry != {"schema_version": 1, "active_pool_capacity": capacity}:
                raise ValueError("Requested inference pool capacity conflicts with the active pool")
        else:
            content = json.dumps({"schema_version": 1, "active_pool_capacity": capacity}).encode("utf-8")
            os.ftruncate(lease, 0)
            os.pwrite(lease, content, 0)
            os.fsync(lease)
        fcntl.flock(lease, fcntl.LOCK_SH)
        return lease
    except BaseException:
        if lease is not None:
            os.close(lease)
        raise
    finally:
        os.close(guard)


def _acquire_inference_locks(lock_path: Path, max_concurrent_inference: int = 1):
    """Acquire coordinated locks before importing CUDA; keep them until exit.

    Legacy/default callers take the original exclusive lock. Pooled callers
    share that same primary lock and each exclusively owns one fixed slot.
    Descriptors intentionally remain open through CUDA interpreter teardown.
    """
    _validate_inference_limit(max_concurrent_inference)
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptors = []
    try:
        primary = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        descriptors.append(primary)
        mode = fcntl.LOCK_EX if max_concurrent_inference == 1 else fcntl.LOCK_SH
        fcntl.flock(primary, mode)
        slot_index, slot_path = None, None
        if max_concurrent_inference > 1:
            descriptors.append(_acquire_pool_capacity(lock_path, max_concurrent_inference))
            while slot_index is None:
                for index in range(max_concurrent_inference):
                    candidate = Path(f"{lock_path}.slot{index}")
                    descriptor = os.open(candidate, os.O_CREAT | os.O_RDWR, 0o600)
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        os.close(descriptor)
                    else:
                        descriptors.append(descriptor)
                        slot_index, slot_path = index, candidate
                        break
                if slot_index is None:
                    time.sleep(.05)
        return descriptors, {
            "primary_path": str(lock_path), "primary_mode": "exclusive" if max_concurrent_inference == 1 else "shared",
            "max_concurrent_inference": max_concurrent_inference,
            "requested_max_concurrent_inference": max_concurrent_inference,
            "active_pool_capacity": max_concurrent_inference,
            "capacity_lease_path": str(Path(f"{lock_path}.capacity")) if max_concurrent_inference > 1 else None,
            "slot_index": slot_index, "slot_path": str(slot_path) if slot_path is not None else None,
            "release_policy": "Kernel releases locks after the isolated CUDA child exits",
        }
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _child_main(event_descriptor: int) -> int:
    os.set_inheritable(event_descriptor, False)
    try:
        payload = json.load(sys.stdin)
        lock_path = Path(payload["lock_path"])
        descriptors, concurrency = _acquire_inference_locks(lock_path, payload.get("max_concurrent_inference", 1))
        # Do not close descriptor or explicitly unlock: the kernel releases it
        # only after interpreter teardown, when CUDA allocations are gone.
        _event(event_descriptor, "inference_running", details={"pid": os.getpid(), **concurrency})
        torch, policy = _configure_deterministic_inference()
        policy["max_concurrent_inference"] = concurrency["max_concurrent_inference"]
        policy["inference_concurrency"] = concurrency
        from totalsegmentator.python_api import totalsegmentator

        totalsegmentator(
            input=Path(payload["image_path"]), output=Path(payload["output"]),
            report=Path(payload["report"]), device=payload["device"],
            quiet=payload["quiet"], nr_thr_saving=2, nr_thr_resamp=1,
            **payload["parameters"],
        )
        _record_execution_policy(Path(payload["report"]), torch, policy)
        return 0
    except Exception as error:
        _event(event_descriptor, "error", error=f"{type(error).__name__}: {error}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        os.close(event_descriptor)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--child":
        raise SystemExit("This helper is launched by the segmentation wrappers")
    raise SystemExit(_child_main(int(sys.argv[2])))
