import csv
import ctypes
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
DLL_PATH = SCRIPT_DIR / "rms_norm_cuda.dll"
CSV_PATH = SCRIPT_DIR / "rms_benchmark_summary.csv"
TXT_PATH = SCRIPT_DIR / "benchmark_results.txt"
PNG_PATH = SCRIPT_DIR / "rms_benchmark_summary.png"

EPS = 1e-6
DTYPE = torch.float32
WARMUP = 50
ITERS = 200

# Key test parameters requested for report.
BATCH_SIZE = 4
SEQ_LEN = 512
DIM = 4096


def _load_library() -> ctypes.CDLL:
    if not DLL_PATH.exists():
        raise FileNotFoundError(f"Cannot find DLL: {DLL_PATH}")
    lib = ctypes.CDLL(str(DLL_PATH))
    argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.launch_rms_norm.argtypes = argtypes
    lib.launch_rms_norm_shared.argtypes = argtypes
    return lib


LIB = _load_library()


def rms_norm_torch(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * rms * w


def rms_norm_cuda(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    batch_size, seq_len, dim = x.shape
    y = torch.empty_like(x)
    LIB.launch_rms_norm(
        x.data_ptr(),
        y.data_ptr(),
        w.data_ptr(),
        eps,
        batch_size,
        seq_len,
        dim,
    )
    return y


def rms_norm_cuda_shared(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    batch_size, seq_len, dim = x.shape
    y = torch.empty_like(x)
    LIB.launch_rms_norm_shared(
        x.data_ptr(),
        y.data_ptr(),
        w.data_ptr(),
        eps,
        batch_size,
        seq_len,
        dim,
    )
    return y


def measure_ms(fn, warmup: int = WARMUP, iters: int = ITERS) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def format_table(rows):
    headers = ["Provider", "Avg Time (ms)", "Max Error", "Speedup(vs PyTorch)", "Throughput (GEl/s)"]
    line = "-" * 110
    out = []
    out.append(line)
    out.append(f"{headers[0]:<16} {headers[1]:>15} {headers[2]:>16} {headers[3]:>24} {headers[4]:>22}")
    out.append(line)
    for row in rows:
        out.append(
            f"{row['provider']:<16} "
            f"{row['avg_ms']:>15.4f} "
            f"{row['max_error']:>16.2e} "
            f"{row['speedup']:>24.2f} "
            f"{row['throughput_gels']:>22.2f}"
        )
    out.append(line)
    return "\n".join(out)


def save_csv(rows):
    target = CSV_PATH
    try:
        f = open(target, "w", newline="", encoding="utf-8")
    except PermissionError:
        target = CSV_PATH.with_name(f"{CSV_PATH.stem}_latest{CSV_PATH.suffix}")
        f = open(target, "w", newline="", encoding="utf-8")
    with f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "provider",
                "avg_ms",
                "max_error",
                "speedup_vs_pytorch",
                "throughput_gels",
                "batch_size",
                "seq_len",
                "dim",
                "dtype",
                "warmup",
                "iters",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "provider": row["provider"],
                    "avg_ms": row["avg_ms"],
                    "max_error": row["max_error"],
                    "speedup_vs_pytorch": row["speedup"],
                    "throughput_gels": row["throughput_gels"],
                    "batch_size": BATCH_SIZE,
                    "seq_len": SEQ_LEN,
                    "dim": DIM,
                    "dtype": str(DTYPE).replace("torch.", ""),
                    "warmup": WARMUP,
                    "iters": ITERS,
                }
            )
    return target


def save_text_report(rows):
    target = TXT_PATH
    try:
        f = open(target, "w", encoding="utf-8")
    except PermissionError:
        target = TXT_PATH.with_name(f"{TXT_PATH.stem}_latest{TXT_PATH.suffix}")
        f = open(target, "w", encoding="utf-8")
    with f:
        f.write("RMS Benchmark Summary\n")
        f.write(f"Device: {torch.cuda.get_device_name(0)}\n")
        f.write(
            f"Config: batch={BATCH_SIZE}, seq={SEQ_LEN}, dim={DIM}, dtype=float32, "
            f"warmup={WARMUP}, iters={ITERS}\n"
        )
        f.write(format_table(rows))
        f.write("\n")
    return target


def save_plot(rows):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Warn] matplotlib unavailable, skip plot: {exc}")
        return None

    providers = [r["provider"] for r in rows]
    times = [r["avg_ms"] for r in rows]
    throughput = [r["throughput_gels"] for r in rows]
    speedups = [r["speedup"] for r in rows]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        (
            "RMS Benchmark\n"
            f"shape=({BATCH_SIZE}, {SEQ_LEN}, {DIM}), dtype=float32, "
            f"warmup={WARMUP}, iters={ITERS}"
        ),
        fontsize=12,
    )

    axes[0].bar(providers, times, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0].set_title("Avg Time (ms)")
    axes[0].set_ylabel("ms")

    axes[1].bar(providers, throughput, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[1].set_title("Throughput (GElements/s)")
    axes[1].set_ylabel("GEl/s")

    axes[2].bar(providers, speedups, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[2].set_title("Speedup vs PyTorch")
    axes[2].set_ylabel("x")

    for i, spd in enumerate(speedups):
        axes[2].text(i, spd, f"{spd:.2f}x", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    target = PNG_PATH
    try:
        fig.savefig(target, dpi=220, bbox_inches="tight")
    except PermissionError:
        target = PNG_PATH.with_name(f"{PNG_PATH.stem}_latest{PNG_PATH.suffix}")
        fig.savefig(target, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return target


def benchmark():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    x = torch.randn(BATCH_SIZE, SEQ_LEN, DIM, device="cuda", dtype=DTYPE)
    w = torch.randn(DIM, device="cuda", dtype=DTYPE)

    providers = {
        "PyTorch": lambda: rms_norm_torch(x, w),
        "CUDA": lambda: rms_norm_cuda(x, w),
        "CUDA Shared": lambda: rms_norm_cuda_shared(x, w),
    }

    baseline = providers["PyTorch"]()
    torch.cuda.synchronize()

    ms = {}
    outputs = {}
    for name, fn in providers.items():
        ms[name] = measure_ms(fn)
        outputs[name] = fn()
    torch.cuda.synchronize()

    pytorch_ms = ms["PyTorch"]
    total_elements = x.numel()

    rows = []
    for name in ["PyTorch", "CUDA", "CUDA Shared"]:
        err = torch.max(torch.abs(outputs[name] - baseline)).item()
        throughput = (total_elements / (ms[name] / 1000.0)) / 1e9
        rows.append(
            {
                "provider": name,
                "avg_ms": ms[name],
                "max_error": err,
                "speedup": pytorch_ms / ms[name],
                "throughput_gels": throughput,
            }
        )

    print("\nRMS Benchmark")
    print(
        f"Device={torch.cuda.get_device_name(0)}, "
        f"shape=({BATCH_SIZE}, {SEQ_LEN}, {DIM}), dtype=float32, warmup={WARMUP}, iters={ITERS}"
    )
    print(format_table(rows))

    csv_out = save_csv(rows)
    txt_out = save_text_report(rows)
    png_out = save_plot(rows)

    print(f"\nSaved: {csv_out}")
    print(f"Saved: {txt_out}")
    if png_out is not None:
        print(f"Saved: {png_out}")
    return rows


if __name__ == "__main__":
    benchmark()
