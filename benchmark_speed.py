import time
import torch
import numpy as np
from models import get_model


def count_parameters(model):
    """Calculates total trainable parameters in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def measure_latency(model, dummy_input, num_warmup=10, num_runs=50):
    """Measures average CPU inference latency in milliseconds and FPS."""
    model.eval()
    
    # Warmup runs to stabilize CPU cache & allocation
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Benchmark runs
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(dummy_input)
            end_time = time.perf_counter()
            timings.append((end_time - start_time) * 1000.0)  # Convert to ms

    avg_latency_ms = float(np.mean(timings))
    std_latency_ms = float(np.std(timings))
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    return avg_latency_ms, std_latency_ms, fps


def main():
    print("=" * 65)
    print("      COCKPIT-AI: COMPUTATIONAL EFFICIENCY BENCHMARK      ")
    print("=" * 65)

    device = torch.device("cpu")
    # Standard 256x256 single-frame input tensor
    dummy_input = torch.randn(1, 3, 256, 256).to(device)

    models_config = [
        ("AOD-Net", "aodnet"),
        ("FFA-Net", "ffanet"),
        ("Mamba", "mamba"),
        ("DehazeTransformer", "transformer"),
    ]

    results = []

    for display_name, model_key in models_config:
        print(f"\n--> Benchmarking {display_name}...")
        try:
            model = get_model(model_key).to(device)
            params_m = count_parameters(model)
            avg_ms, std_ms, fps = measure_latency(model, dummy_input)
            
            results.append({
                "Model": display_name,
                "Params (M)": f"{params_m:.3f}M",
                "Latency (ms)": f"{avg_ms:.2f} ± {std_ms:.2f}",
                "FPS": f"{fps:.1f}"
            })
            print(f"    Params: {params_m:.3f}M | Latency: {avg_ms:.2f} ms | FPS: {fps:.1f}")
        except Exception as e:
            print(f"    Error benchmarking {display_name}: {e}")

    print("\n" + "=" * 65)
    print(f"{'Model Architecture':<20} | {'Params (M)':<12} | {'Latency (ms)':<15} | {'FPS':<8}")
    print("-" * 65)
    for res in results:
        print(f"{res['Model']:<20} | {res['Params (M)']:<12} | {res['Latency (ms)']:<15} | {res['FPS']:<8}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()