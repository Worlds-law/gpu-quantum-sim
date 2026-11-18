import cudaq
import time
import statistics

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
    
    '''
    hadamard to the first qubit then |psi1> = 1/sqrt(2) (|00> + |10>) 
    after the CNOT |psi2>  = 1/sqrt(2) (|00> + |11>) which is the GHZ state
    
    removing CNOT still gives similar numbers of 00 and 11 since hadamard alone produces 2 outcomes w 50% probability each. 
    '''
    
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

    # 3: Calculate the median and standard deviation.
    if runs < 2:
        raise ValueError("At least two timed runs are required.")
    median_time = statistics.median(times)
    std_dev_time = statistics.stdev(times)
    #  4: Return both statistics.
    return median_time, std_dev_time

def fits_memory(
    n: int,
    available_bytes: int,
    safety_fraction: float = 0.5
) -> bool:
    """
    Check whether an n-qubit FP64 statevector fits
    within a conservative memory budget.
    """
    if n < 0:
        raise ValueError("Number of qubits must be non-negative.")
    if not (0 < safety_fraction <= 1):
        raise ValueError("Safety fraction must be in the range (0, 1].")

    estimated_memory = estimate_memory(n, precision="fp64")
    return estimated_memory <= available_bytes * safety_fraction


if __name__ == "__main__":
    for n in [8, 12, 16]:
        median, std = benchmark_ghz(n, runs=5)
        print(f"n={n}: median={median:.6f}s, std={std:.6f}s")
    median, std = benchmark_ghz(8, runs=5)
    counts = cudaq.sample(ghz, 4, shots_count=1000)
    print(counts)
    