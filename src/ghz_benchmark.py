import cudaq
import time
import statistics
import csv
from pathlib import Path
import psutil
import matplotlib.pyplot as plt
import argparse

cudaq.set_target("qpp-cpu")



@cudaq.kernel
def ghz(n: int):
    q = cudaq.qvector(n)

    h(q[0])

    #1: Entangle adjacent qubits using a chain of CNOT gates
    for i in range(n - 1):
        x.ctrl(q[i], q[i + 1])

    #2: Measure all qubits
    mz(q)
    
    
def estimate_memory(n: int, precision: str = "fp64") -> int:
    """
    Estimate the memory usage of the GHZ kernel for a given number of qubits.

    Args:
        n (int): The number of qubits.
        precision (str): The precision of the simulation. "fp32" or "fp64".

    Returns:
        Estimated memory usage in bytes.
    """
    if precision == "fp32":
        bytes_per_complex = 8  # 4 bytes for real + 4 bytes for imaginary
    elif precision == "fp64":
        bytes_per_complex = 16  # 8 bytes for real + 8 bytes for imaginary
    else:
        raise ValueError("Precision must be 'fp32' or 'fp64'.")
    
    if n < 0:
        raise ValueError("Number of qubits must be non-negative.")

    memory = (2 ** n) * bytes_per_complex
    return memory
    
    
def benchmark_ghz(n: int, runs: int = 5):
    """
    Benchmark GHZ circuit execution.

    Returns:
        Median runtime and standard deviation in seconds.
    """
    if runs < 2:
        raise ValueError("At least two timed runs are required.")
    #1: Run the GHZ circuit once as a warm-up.
    cudaq.sample(ghz, n, shots_count=1000)
    
    times = []

    #2: Repeat the following 'runs' times:
    #   Record the starting time
    #   Execute cudaq.sample()
    #   Record the elapsed time
    #   Append the elapsed time to 'times'
    for _ in range(runs):
        start_time = time.perf_counter()
        cudaq.sample(ghz, n, shots_count=1000)
        elapsed_time = time.perf_counter() - start_time
        times.append(elapsed_time)

    #3: Calculate the median and standard deviation.
    median_time = statistics.median(times)
    std_dev_time = statistics.stdev(times)
    
    return median_time, std_dev_time

def fits_memory(
    n: int,
    available_bytes: int,
    safety_fraction: float = 0.5
) -> bool:
    """
    Check whether an n-qubit FP64 statevector fits
    """
    if n < 0:
        raise ValueError("Number of qubits must be non-negative.")
    if not (0 < safety_fraction <= 1):
        raise ValueError("Safety fraction must be in the range (0, 1].")

    estimated_memory = estimate_memory(n, precision="fp64")
    return estimated_memory <= available_bytes * safety_fraction

def get_available_memory() -> int:
    """
    Return estimated available system RAM in bytes.
    """
    memory = psutil.virtual_memory()
    return memory.available

def run_benchmarks(
    min_qubits: int = 8,
    max_qubits: int = 16,
    runs: int = 5
):
    """
    Benchmark GHZ circuits over a range of qubit counts.
    Skip configurations that exceed the memory budget.
    """
    results = []

    for n in range(min_qubits, max_qubits + 1):
        available = get_available_memory()

        if not fits_memory(n, available):
            print(f"{n} qubits: SKIP")
            continue

        median, std = benchmark_ghz(n, runs)

    
        results.append({
            "n": n,
            "median_s": median,
            "std_s": std,
            "est_mem_gib": estimate_memory(n, precision="fp64") / (1024**3)
        })

    return results

def save_results(results: list[dict], filename: str) -> None:
    parent_dir = Path(filename).parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    with open(filename, mode='w', newline='') as csvfile:
        fieldnames = ["n", "median_s", "std_s", "est_mem_gib"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)
    
    
    
def plot_benchmarks(results: list[dict], filename: str) -> None:
        n_values = [result["n"] for result in results]

        median_runtimes = [result["median_s"] for result in results]

        plt.figure(figsize=(10, 6))
    
        plt.plot(n_values, median_runtimes, marker='o', linestyle='-', color='b', label='Median Runtime')
        plt.yscale('log')
        plt.xlabel("Number of Qubits")
        plt.ylabel("Median Runtime (s)")
        plt.title("GHZ State Simulation Performance")
        plt.grid(True)
        plt.savefig(filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Benchmark GHZ state simulation using CUDA-Q"
)
    parser.add_argument("--min-qubits", type=int, default=8, help="Minimum number of qubits to simulate")
    parser.add_argument("--max-qubits", type=int, default=16, help="Maximum number of qubits to simulate")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs for each simulation")
    args = parser.parse_args()
    if args.min_qubits < 1:
        parser.error("--min-qubits must be at least 1")

    if args.max_qubits < args.min_qubits:
        parser.error("--max-qubits must be >= --min-qubits")

    if args.runs < 2:
        parser.error("--runs must be at least 2")
        
   
    
    results = run_benchmarks(
        min_qubits=args.min_qubits,
        max_qubits=args.max_qubits,
        runs=args.runs
    )
    
    save_results(results, "results/ghz_benchmark_results.csv")

    for result in results:
        print(
            f"n={result['n']}: "
            f"median={result['median_s']:.6f}s, "
            f"std={result['std_s']:.6f}s"
        )
    
    plot_benchmarks(results, "results/ghz_benchmark_plot.png")
    