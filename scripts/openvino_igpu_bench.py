#!/usr/bin/env python3
"""Go/no-go benchmark: is the Intel iGPU faster than the CPU under OpenVINO?

Builds a synthetic transformer-shaped stack (sizes of a ~0.6B model: hidden 1024,
MLP 3072, 28 layers, no attention-score math) and times it on every OpenVINO
device. Two shapes matter for TTS:

  decode   1 token per step (autoregressive generation; memory-bandwidth bound)
  prefill  64 tokens at once (prompt processing; compute bound)

Needs only `pip install openvino numpy`. Run it in a venv, not the system Python
(Arch's python-openvino package has no GPU plugin).

    python scripts/openvino_igpu_bench.py
    python scripts/openvino_igpu_bench.py --layers 8      # quicker smoke test
"""

import argparse
import statistics
import time

import numpy as np
import openvino as ov
import openvino.opset13 as op

HIDDEN, MLP = 1024, 3072


def build_model(layers: int, seq: int) -> ov.Model:
    rng = np.random.default_rng(0)

    def w(rows: int, cols: int):
        return op.constant((rng.standard_normal((rows, cols)) * 0.02).astype(np.float16))

    def linear(x, rows, cols):
        # Weights stored fp16 and converted, as a real fp16 deployment would.
        return op.matmul(x, op.convert(w(rows, cols), np.float32), False, False)

    inp = op.parameter([1, seq, HIDDEN], np.float32, name="x")
    x = inp
    for _ in range(layers):
        # attention-shaped projections (q, k, v, o); no softmax/KV cache here
        q, k, v = (linear(x, HIDDEN, HIDDEN) for _ in range(3))
        a = linear(op.add(op.add(q, k), v), HIDDEN, HIDDEN)
        x = op.add(x, a)
        # gated MLP
        g = op.swish(linear(x, HIDDEN, MLP))
        u = linear(x, HIDDEN, MLP)
        x = op.add(x, linear(op.multiply(g, u), MLP, HIDDEN))
    return ov.Model([x], [inp], f"bench_{seq}")


def bench(core: ov.Core, device: str, model: ov.Model, seq: int, iters: int) -> float:
    prec = "f16" if device.startswith("GPU") else "f32"
    compiled = core.compile_model(
        model, device, {"INFERENCE_PRECISION_HINT": prec, "PERFORMANCE_HINT": "LATENCY"}
    )
    x = np.random.default_rng(1).standard_normal((1, seq, HIDDEN)).astype(np.float32)
    req = compiled.create_infer_request()
    for _ in range(3):  # warmup (kernel compile, caches)
        req.infer({0: x})
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        req.infer({0: x})
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--iters", type=int, default=15)
    args = ap.parse_args()

    core = ov.Core()
    devices = [d for d in core.available_devices if d.startswith(("CPU", "GPU"))]
    print(f"OpenVINO {ov.get_version()}")
    for d in devices:
        print(f"  {d}: {core.get_property(d, 'FULL_DEVICE_NAME')}")
    if not any(d.startswith("GPU") for d in devices):
        print("\nNo GPU device visible to OpenVINO; nothing to compare.")

    results: dict[tuple[str, str], float] = {}
    for label, seq in (("decode (1 token)", 1), ("prefill (64 tokens)", 64)):
        model = build_model(args.layers, seq)
        for d in devices:
            try:
                results[(label, d)] = bench(core, d, model, seq, args.iters)
            except Exception as e:  # keep going; a failing device is itself a result
                print(f"\n[{d}] {label}: FAILED: {str(e).splitlines()[0][:200]}")

    print(f"\nMedian latency per forward pass ({args.layers} layers), lower is better\n")
    print(f"{'workload':<22}{'device':<10}{'ms':>10}{'vs CPU':>10}")
    for label in ("decode (1 token)", "prefill (64 tokens)"):
        cpu = results.get((label, "CPU"))
        for d in devices:
            ms = results.get((label, d))
            if ms is None:
                continue
            rel = f"{cpu / ms:.2f}x" if cpu and d != "CPU" else ""
            print(f"{label:<22}{d:<10}{ms:>10.1f}{rel:>10}")
        print()
    print("'vs CPU' above 1.00x means the GPU is faster. Roughly: decode matters most for")
    print("TTS speed. If GPU decode is not clearly above 1.5x, OpenVINO on this iGPU is")
    print("probably not worth wiring into Voicebox.")


if __name__ == "__main__":
    main()
