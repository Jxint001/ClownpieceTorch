"""
Test Part 3: Parameter Initialization
Tests the initialization functions in nn.init including:
- Basic initializations (zeros_, ones_, constants_, uniform_, normal_)
- Advanced initializations (xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_)
- Statistical properties of initialized tensors
"""

from graderlib import set_debug_mode, testcase, grader_summary, tensor_close
import clownpiece as CP
from clownpiece import Tensor
from clownpiece.nn import init
from clownpiece.autograd import no_grad
import numpy as np
import math

@testcase(name="init_constants", score=10)
def test_init_constants():
    """Test constant initialization functions"""
    tensor = Tensor.empty((3, 4))
    
    # Test zeros_
    init.zeros_(tensor)
    expected_zeros = Tensor.zeros((3, 4))
    assert tensor_close(tensor, expected_zeros), \
        f"zeros_ failed. Expected all zeros, got {tensor.tolist()}"
    
    # Test ones_
    init.ones_(tensor)
    expected_ones = Tensor.ones((3, 4))
    assert tensor_close(tensor, expected_ones), \
        f"ones_ failed. Expected all ones, got {tensor.tolist()}"
    
    # Test constant_
    init.constants_(tensor, 3.14)
    expected_const = Tensor.ones((3, 4)) * 3.14
    assert tensor_close(tensor, expected_const), \
        f"constant_ failed. Expected all 3.14, got {tensor.tolist()}"
    
    return True

@testcase(name="init_uniform", score=10)
def test_init_uniform():
    """Test uniform initialization"""
    tensor = Tensor.empty((1000, 10))  # Large tensor for statistical testing
    
    # Initialize with uniform distribution
    low, high = -2.0, 3.0
    init.uniform_(tensor, low, high)
    
    # Check range
    arr = np.array(tensor.tolist())
    assert np.all(arr >= low) and np.all(arr <= high), \
        f"uniform_ values outside range [{low}, {high}]. Min: {arr.min()}, Max: {arr.max()}"
    
    # Check approximate statistics (for large tensor)
    mean = np.mean(arr)
    expected_mean = (low + high) / 2
    assert abs(mean - expected_mean) < 0.1, \
        f"uniform_ mean {mean} too far from expected {expected_mean}"
    
    return True

@testcase(name="init_normal", score=10)
def test_init_normal():
    """Test normal initialization"""
    tensor = Tensor.empty((1000, 10))  # Large tensor for statistical testing
    
    # Initialize with normal distribution
    mean, std = 1.0, 0.5
    init.normal_(tensor, mean, std)
    
    # Check approximate statistics (for large tensor)
    arr = np.array(tensor.tolist())
    actual_mean = np.mean(arr)
    actual_std = np.std(arr)
    
    assert abs(actual_mean - mean) < 0.1, \
        f"normal_ mean {actual_mean} too far from expected {mean}"
    assert abs(actual_std - std) < 0.1, \
        f"normal_ std {actual_std} too far from expected {std}"
    
    return True

@testcase(name="init_xavier_uniform", score=10)
def test_init_xavier_uniform():
    """Test Xavier uniform initialization"""
    # Test with 10x20 tensors as mentioned in documentation
    in_features, out_features = 10, 20
    
    # Run test 1000 times as mentioned in documentation
    num_tests = 1000
    all_values = []
    
    for _ in range(num_tests):
        tensor = Tensor.empty((out_features, in_features))
        gain = 1.0
        init.xavier_uniform_(tensor, gain)
        arr = np.array(tensor.tolist())
        all_values.extend(arr.flatten())
    
    # Check expected variance
    fan_in = in_features
    fan_out = out_features
    expected_std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    expected_bound = expected_std * math.sqrt(3.0)  # For uniform distribution
    
    all_values = np.array(all_values)
    
    # Check that all values are within bounds
    assert np.all(np.abs(all_values) <= expected_bound * 1.1), \
        f"xavier_uniform_ values outside expected bound ±{expected_bound}"
    
    # Check approximate variance across all runs
    actual_var = np.var(all_values)
    expected_var = expected_std ** 2
    assert abs(actual_var - expected_var) < expected_var * 0.3, \
        f"xavier_uniform_ variance {actual_var} too far from expected {expected_var}"
    
    # Check that mean is close to 0
    actual_mean = np.mean(all_values)
    assert abs(actual_mean) < 0.1, \
        f"xavier_uniform_ mean {actual_mean} should be close to 0"
    
    return True

@testcase(name="init_xavier_normal", score=10)
def test_init_xavier_normal():
    """Test Xavier normal initialization"""
    # Test with 10x20 tensors as mentioned in documentation
    in_features, out_features = 10, 20
    
    # Run test 1000 times as mentioned in documentation
    num_tests = 1000
    all_values = []
    
    for _ in range(num_tests):
        tensor = Tensor.empty((out_features, in_features))
        gain = 1.0
        init.xavier_normal_(tensor, gain)
        arr = np.array(tensor.tolist())
        all_values.extend(arr.flatten())
    
    # Check expected standard deviation
    fan_in = in_features
    fan_out = out_features
    expected_std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    
    all_values = np.array(all_values)
    
    # Check approximate statistics across all runs
    actual_std = np.std(all_values)
    assert abs(actual_std - expected_std) < expected_std * 0.3, \
        f"xavier_normal_ std {actual_std} too far from expected {expected_std}"
    
    # Check that mean is close to 0
    actual_mean = np.mean(all_values)
    assert abs(actual_mean) < 0.1, \
        f"xavier_normal_ mean {actual_mean} should be close to 0"
    
    return True

@testcase(name="init_kaiming_uniform", score=10)
def test_init_kaiming_uniform():
    """Test Kaiming uniform initialization"""
    # Test with 10x20 tensors as mentioned in documentation
    in_features, out_features = 10, 20
    
    # Run test 1000 times as mentioned in documentation
    num_tests = 1000
    all_values = []
    
    for _ in range(num_tests):
        tensor = Tensor.empty((out_features, in_features))
        # Test fan_in mode
        init.kaiming_uniform_(tensor, a=0, mode="fan_in", nonlinearity="relu")
        arr = np.array(tensor.tolist())
        all_values.extend(arr.flatten())
    
    # Check expected variance
    fan_in = in_features
    gain = init.calculate_gain("relu")
    expected_std = gain / math.sqrt(fan_in)
    expected_bound = expected_std * math.sqrt(3.0)  # For uniform distribution
    
    all_values = np.array(all_values)
    
    # Check that all values are within bounds
    assert np.all(np.abs(all_values) <= expected_bound * 1.1), \
        f"kaiming_uniform_ values outside expected bound ±{expected_bound}"
    
    # Check approximate variance across all runs
    actual_var = np.var(all_values)
    expected_var = expected_std ** 2
    assert abs(actual_var - expected_var) < expected_var * 0.4, \
        f"kaiming_uniform_ variance {actual_var} too far from expected {expected_var}"
    
    # Check that mean is close to 0
    actual_mean = np.mean(all_values)
    assert abs(actual_mean) < 0.1, \
        f"kaiming_uniform_ mean {actual_mean} should be close to 0"
    
    return True

@testcase(name="init_kaiming_normal", score=10)
def test_init_kaiming_normal():
    """Test Kaiming normal initialization"""
    # Test with 10x20 tensors as mentioned in documentation
    in_features, out_features = 10, 20
    
    # Run test 1000 times as mentioned in documentation
    num_tests = 1000
    all_values = []
    
    for _ in range(num_tests):
        tensor = Tensor.empty((out_features, in_features))
        # Test fan_out mode
        init.kaiming_normal_(tensor, a=0, mode="fan_out", nonlinearity="relu")
        arr = np.array(tensor.tolist())
        all_values.extend(arr.flatten())
    
    # Check expected standard deviation
    fan_out = out_features
    gain = init.calculate_gain("relu")
    expected_std = gain / math.sqrt(fan_out)
    
    all_values = np.array(all_values)
    
    # Check approximate statistics across all runs
    actual_std = np.std(all_values)
    assert abs(actual_std - expected_std) < expected_std * 0.3, \
        f"kaiming_normal_ std {actual_std} too far from expected {expected_std}"
    
    # Check that mean is close to 0
    actual_mean = np.mean(all_values)
    assert abs(actual_mean) < 0.1, \
        f"kaiming_normal_ mean {actual_mean} should be close to 0"
    
    return True

@testcase(name="init_with_linear_layer", score=10)
def test_init_with_linear_layer():
    """Test initialization with actual Linear layer"""
    from clownpiece.nn import Linear
    
    # Create Linear layer
    linear = Linear(5, 3, bias=True)
    
    # Test that parameters are initialized
    weight_list = linear.weight.tolist()
    bias_list = linear.bias.tolist()
    
    # Check that they are not all zeros (should be initialized)
    assert not all(all(val == 0 for val in row) for row in weight_list), \
        "Linear weight should be initialized to non-zero values"
    assert not all(val == 0 for val in bias_list), \
        "Linear bias should be initialized to non-zero values"
    
    # Manually reinitialize with xavier
    with no_grad():
        init.xavier_uniform_(linear.weight, gain=1.0)
        init.zeros_(linear.bias)
    
    # Check that manual initialization worked
    new_weight_list = linear.weight.tolist()
    new_bias_list = linear.bias.tolist()
    
    assert weight_list != new_weight_list, \
        "Weight should change after reinitialization"
    assert all(val == 0 for val in new_bias_list), \
        "Bias should be zero after zeros_ initialization"
    
    return True

if __name__ == "__main__":
    print("Testing Week 3 Part 3: Parameter Initialization")
    print("=" * 50)
    
    test_init_constants()
    test_init_uniform()
    test_init_normal()
    test_init_xavier_uniform()
    test_init_xavier_normal()
    test_init_kaiming_uniform()
    test_init_kaiming_normal()
    test_init_with_linear_layer()
    
    grader_summary()
