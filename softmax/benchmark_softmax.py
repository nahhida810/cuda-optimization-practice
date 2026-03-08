import csv
import ctypes
from pathlib import Path
from statistics import mean

import torch

from fused_softmax import triton_fused_softmax
from tiling_softmax import triton_tiling_softmax


SCRIPT_DIR = Path(__file__).resolve().parent
DLL_PATH = SCRIPT_DIR / "softmax_cuda.dll"
SUMMARY_CSV = SCRIPT_DIR / "softmax_benchmark_summary.csv"
DETAIL_CSV = SCRIPT_DIR / "softmax_benchmark_detail.csv"
PNG_PATH = SCRIPT_DIR / "softmax_benchmark_summary.png"
TXT_PATH = SCRIPT_DIR / "softmax_benchmark_report.txt"

DTYPE = torch.float32
WARMUP = 40
ITERS = 200
INPUT_POOL_SIZE = 8
TRITON_MODE = "autotune"
TRITON_AUTOTUNE_KEY = "n_cols"

# Key benchmark parameters for report.
TEST_SHAPES = [
    (4096, 1024),
    (2048, 4096),
    (1024, 8192),
]


def _load_cuda_library() -> ctypes.CDLL:
    if not DLL_PATH.exists():
        raise FileNotFoundError(f"Cannot find DLL: {DLL_PATH}")
    lib = ctypes.CDLL(str(DLL_PATH))
    lib.launch_softmax.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    return lib


LIB = _load_cuda_library()


def pytorch_softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def cuda_softmax(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.float32:
        raise ValueError("CUDA DLL path currently supports float32 only")
    if not x.is_contiguous():
        x = x.contiguous()
    rows, cols = x.shape
    y = torch.empty_like(x)
    LIB.launch_softmax(x.data_ptr(), y.data_ptr(), cols, rows)
    return y


def fused_softmax(x: torch.Tensor) -> torch.Tensor:
    return triton_fused_softmax(x)


def tiling_softmax(x: torch.Tensor) -> torch.Tensor:
    return triton_tiling_softmax(x)


def measure_ms_with_pool(fn, x_pool, warmup: int = WARMUP, iters: int = ITERS) -> float:
    for i in range(warmup):
        fn(x_pool[i % len(x_pool)])
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(iters):
        fn(x_pool[i % len(x_pool)])
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def throughput_gels(rows: int, cols: int, ms: float) -> float:
    elements = rows * cols
    return (elements / (ms / 1000.0)) / 1e9


def format_summary_table(rows):
    headers = ["Provider", "Avg Time (ms)", "Max Error", "Speedup(vs PyTorch)", "Throughput (GEl/s)"]
    line = "-" * 104
    out = []
    out.append(line)
    out.append(f"{headers[0]:<18}{headers[1]:>15}{headers[2]:>16}{headers[3]:>24}{headers[4]:>23}")
    out.append(line)
    for row in rows:
        out.append(
            f"{row['provider']:<18}"
            f"{row['avg_ms']:>15.4f}"
            f"{row['max_error']:>16.2e}"
            f"{row['speedup']:>24.2f}"
            f"{row['throughput_gels']:>23.2f}"
        )
    out.append(line)
    return "\n".join(out)


def save_csv(summary_rows, detail_rows):
    summary_target = SUMMARY_CSV
    try:
        f = open(summary_target, "w", newline="", encoding="utf-8")
    except PermissionError:
        summary_target = SUMMARY_CSV.with_name(f"{SUMMARY_CSV.stem}_latest{SUMMARY_CSV.suffix}")
        f = open(summary_target, "w", newline="", encoding="utf-8")
    with f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "provider",
                "avg_ms",
                "max_error",
                "speedup_vs_pytorch",
                "throughput_gels",
                "dtype",
                "warmup",
                "iters",
                "input_pool_size",
                "test_shapes",
                "triton_mode",
                "triton_autotune_key",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    "provider": row["provider"],
                    "avg_ms": row["avg_ms"],
                    "max_error": row["max_error"],
                    "speedup_vs_pytorch": row["speedup"],
                    "throughput_gels": row["throughput_gels"],
                    "dtype": str(DTYPE).replace("torch.", ""),
                    "warmup": WARMUP,
                    "iters": ITERS,
                    "input_pool_size": INPUT_POOL_SIZE,
                    "test_shapes": "|".join([f"{r}x{c}" for r, c in TEST_SHAPES]),
                    "triton_mode": TRITON_MODE,
                    "triton_autotune_key": TRITON_AUTOTUNE_KEY,
                }
            )

    detail_target = DETAIL_CSV
    try:
        f = open(detail_target, "w", newline="", encoding="utf-8")
    except PermissionError:
        detail_target = DETAIL_CSV.with_name(f"{DETAIL_CSV.stem}_latest{DETAIL_CSV.suffix}")
        f = open(detail_target, "w", newline="", encoding="utf-8")
    with f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "provider",
                "rows",
                "cols",
                "avg_ms",
                "max_error",
                "throughput_gels",
                "elements",
            ],
        )
        writer.writeheader()
        for row in detail_rows:
            writer.writerow(row)
    return summary_target, detail_target


def save_report(summary_rows, detail_rows):
    target = TXT_PATH
    try:
        f = open(target, "w", encoding="utf-8")
    except PermissionError:
        target = TXT_PATH.with_name(f"{TXT_PATH.stem}_latest{TXT_PATH.suffix}")
        f = open(target, "w", encoding="utf-8")
    with f:
        f.write("Softmax Benchmark Report\n")
        f.write(f"Device: {torch.cuda.get_device_name(0)}\n")
        f.write(
            f"Config: dtype=float32, warmup={WARMUP}, iters={ITERS}, input_pool_size={INPUT_POOL_SIZE}, "
            f"triton_mode={TRITON_MODE}, triton_autotune_key={TRITON_AUTOTUNE_KEY}\n"
        )
        f.write(f"Shapes: {', '.join([str(s) for s in TEST_SHAPES])}\n\n")
        f.write("Summary\n")
        f.write(format_summary_table(summary_rows))
        f.write("\n\nDetail\n")
        for row in detail_rows:
            f.write(
                f"{row['provider']:<18} shape=({row['rows']}, {row['cols']}), "
                f"time={row['avg_ms']:.4f} ms, err={row['max_error']:.2e}, "
                f"throughput={row['throughput_gels']:.2f} GEl/s\n"
            )
    return target


def save_plot(summary_rows, detail_rows):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Warn] matplotlib unavailable, skip plot: {exc}")
        return None

    providers = [r["provider"] for r in summary_rows]
    avg_ms = [r["avg_ms"] for r in summary_rows]
    speedups = [r["speedup"] for r in summary_rows]
    throughput = [r["throughput_gels"] for r in summary_rows]

    shape_labels = [f"{r}x{c}" for r, c in TEST_SHAPES]
    detail_map = {
        (r["provider"], f"{r['rows']}x{r['cols']}"): r["avg_ms"] for r in detail_rows
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

    axes[0, 0].bar(providers, avg_ms, color=colors)
    axes[0, 0].set_title("Average Time")
    axes[0, 0].set_ylabel("ms")

    axes[0, 1].bar(providers, speedups, color=colors)
    axes[0, 1].set_title("Speedup vs PyTorch")
    axes[0, 1].set_ylabel("x")

    axes[1, 0].bar(providers, throughput, color=colors)
    axes[1, 0].set_title("Throughput")
    axes[1, 0].set_ylabel("GElements/s")

    for provider, color in zip(providers, colors):
        y = [detail_map[(provider, shape)] for shape in shape_labels]
        axes[1, 1].plot(shape_labels, y, marker="o", label=provider, color=color)
    axes[1, 1].set_title("Latency by Shape")
    axes[1, 1].set_ylabel("ms")
    axes[1, 1].set_xlabel("Shape (rows x cols)")
    axes[1, 1].legend()

    fig.suptitle(
        "Softmax Benchmark\n"
        f"dtype=float32, warmup={WARMUP}, iters={ITERS}, pool={INPUT_POOL_SIZE}, "
        f"shapes={shape_labels}, triton_mode={TRITON_MODE}, key={TRITON_AUTOTUNE_KEY}",
        fontsize=12,
    )
    fig.tight_layout()
    target = PNG_PATH
    try:
        fig.savefig(target, dpi=220, bbox_inches="tight")
    except PermissionError:
        target = PNG_PATH.with_name(f"{PNG_PATH.stem}_latest{PNG_PATH.suffix}")
        fig.savefig(target, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return target


def run_benchmark():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    providers = [
        ("PyTorch CUDA", pytorch_softmax),
        ("CUDA", cuda_softmax),
        ("Triton Fused", fused_softmax),
        ("Triton Tiling", tiling_softmax),
    ]

    detail_rows = []
    provider_to_times = {name: [] for name, _ in providers}
    provider_to_errors = {name: [] for name, _ in providers}
    provider_to_throughput = {name: [] for name, _ in providers}

    for rows, cols in TEST_SHAPES:
        if cols % 4 != 0:
            raise ValueError(f"cols must be divisible by 4 for CUDA kernel, got cols={cols}")

        x_pool = [torch.randn(rows, cols, device="cuda", dtype=DTYPE) for _ in range(INPUT_POOL_SIZE)]
        baseline = pytorch_softmax(x_pool[0])

        for name, fn in providers:
            ms = measure_ms_with_pool(fn, x_pool)
            y = fn(x_pool[0])
            err = torch.max(torch.abs(y - baseline)).item()
            thpt = throughput_gels(rows, cols, ms)

            detail_rows.append(
                {
                    "provider": name,
                    "rows": rows,
                    "cols": cols,
                    "avg_ms": ms,
                    "max_error": err,
                    "throughput_gels": thpt,
                    "elements": rows * cols,
                }
            )
            provider_to_times[name].append(ms)
            provider_to_errors[name].append(err)
            provider_to_throughput[name].append(thpt)

    pytorch_avg = mean(provider_to_times["PyTorch CUDA"])

    summary_rows = []
    for name, _ in providers:
        avg_ms = mean(provider_to_times[name])
        max_error = max(provider_to_errors[name])
        speedup = pytorch_avg / avg_ms
        avg_thpt = mean(provider_to_throughput[name])
        summary_rows.append(
            {
                "provider": name,
                "avg_ms": avg_ms,
                "max_error": max_error,
                "speedup": speedup,
                "throughput_gels": avg_thpt,
            }
        )

    print("\nSoftmax Benchmark")
    print(
        f"Device={torch.cuda.get_device_name(0)}, dtype=float32, warmup={WARMUP}, "
        f"iters={ITERS}, pool={INPUT_POOL_SIZE}, triton_mode={TRITON_MODE}"
    )
    print(f"Triton autotune key={TRITON_AUTOTUNE_KEY}")
    print(f"Shapes={TEST_SHAPES}")
    print(format_summary_table(summary_rows))

    summary_out, detail_out = save_csv(summary_rows, detail_rows)
    txt_out = save_report(summary_rows, detail_rows)
    png_out = save_plot(summary_rows, detail_rows)

    print(f"\nSaved: {summary_out}")
    print(f"Saved: {detail_out}")
    print(f"Saved: {txt_out}")
    if png_out is not None:
        print(f"Saved: {png_out}")
    return summary_rows, detail_rows


if __name__ == "__main__":
    run_benchmark()
