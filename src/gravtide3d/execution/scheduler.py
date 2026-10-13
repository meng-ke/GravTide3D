from __future__ import annotations


def describe_plan(plan) -> str:
    dev = "CPU" if plan.backend == "cpu" else ",".join(f"GPU {d}" for d in plan.devices)
    return (
        f"backend={plan.backend} ({dev}); memory={plan.mode}; dtype={plan.dtype}; "
        f"estimated_peak={plan.estimated_peak_bytes} bytes; reason={plan.reason}"
    )
