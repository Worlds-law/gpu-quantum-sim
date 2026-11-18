import cudaq
import pytest
from src.ghz_benchmark import estimate_memory, ghz, fits_memory

def test_ghz():
    counts = cudaq.sample(ghz, 4, shots_count=1000)

    #Check that only 0000 and 1111 appear.
    allowed = {"0000", "1111"}
    observed = set(counts)
    assert observed.issubset(allowed), f"Unexpected measurement outcomes found: {observed - allowed}"

    #Calculate the fraction of 0000 outcomes.
    fraction_0000 = counts.count("0000") / counts.get_total_shots()

    #Assert that this fraction is close to 0.5.
    assert abs(fraction_0000 - 0.5) < 0.1
    
def test_memory_estimator():
    assert estimate_memory(20, "fp64") == 16 * 1024**2
    assert estimate_memory(20, "fp32") == 8 * 1024**2
    assert estimate_memory(30, "fp64") == 16 * 1024**3
    assert estimate_memory(0, "fp64") == 16
    with pytest.raises(ValueError):
        estimate_memory(10, "fp16")
    with pytest.raises(ValueError):
        estimate_memory(-1, "fp64")
    
    
def test_fits_memory():
    available = 8 * 1024**3

    assert fits_memory(24, available, safety_fraction=0.5) == True
    assert fits_memory(28, available, safety_fraction=0.5) == True
    assert fits_memory(29, available, safety_fraction=0.5) == False