"""
Test Part 2: Simplest Concrete Modules
Tests the simplest concrete modules as defined in the tutorial:
- Linear layer (affine transformation)
- Tanh activation function
"""

from graderlib import set_debug_mode, testcase, grader_summary, tensor_close
import clownpiece as CP
from clownpiece import Tensor
from clownpiece.nn import Module, Linear, Tanh
from clownpiece.autograd import no_grad
import math

@testcase(name="tanh_forward", score=10)
def test_tanh_forward():
    """Test Tanh activation with known values"""
    tanh = Tanh()
    
    # Test with known values
    input_tensor = Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    output = tanh(input_tensor)
    
    # Expected: tanh(-2), tanh(-1), tanh(0), tanh(1), tanh(2)
    expected = Tensor([[-0.9640, -0.7616, 0.0, 0.7616, 0.9640]])
    
    assert tensor_close(output, expected, rtol=1e-3), \
        f"Expected {expected.tolist()}, got {output.tolist()}"
    
    return True

@testcase(name="tanh_module_properties", score=10)
def test_tanh_module_properties():
    """Test that Tanh module has correct properties"""
    tanh = Tanh()
    
    # Test that it's a Module
    assert isinstance(tanh, Module), "Tanh should inherit from Module"
    
    # Test that it has no parameters (activation functions are stateless)
    assert len(list(tanh.parameters())) == 0, "Tanh should have no parameters"
    
    # Test training mode
    assert tanh.training == True, "Tanh should be in training mode by default"
    tanh.eval()
    assert tanh.training == False, "Tanh should be in eval mode after calling eval()"
    
    return True

@testcase(name="tanh_batch_processing", score=10)
def test_tanh_batch_processing():
    """Test Tanh with batch inputs"""
    tanh = Tanh()
    
    batch_input = Tensor([
        [-1.0, 0.0, 1.0],
        [2.0, -2.0, 0.5],
        [-0.5, 1.5, -1.5]
    ])
    
    output = tanh(batch_input)
    
    # Check shape preservation
    assert output.shape == batch_input.shape, \
        f"Tanh output shape {output.shape} doesn't match input shape {batch_input.shape}"
    
    # Check that all values are between -1 and 1
    output_list = output.tolist()
    assert all(all(-1 <= val <= 1 for val in row) for row in output_list), \
        "Tanh output should be between -1 and 1"
    
    return True

@testcase(name="tanh_repr", score=10)
def test_tanh_repr():
    """Test string representation of Tanh module"""
    tanh = Tanh()
    
    # Test that repr contains class name
    tanh_repr = str(tanh)
    assert "Tanh" in tanh_repr, f"Tanh repr should contain 'Tanh', got: {tanh_repr}"
    
    print(f"Tanh repr: {tanh_repr}")
    
    return True

@testcase(name="linear_forward_deterministic", score=10)
def test_linear_forward_deterministic():
    """Test Linear layer with predetermined weights and bias"""
    # Create Linear layer
    linear = Linear(3, 2, bias=True)
    
    # Set specific weights and bias
    with no_grad():
        # weight shape: (out_features, in_features) = (2, 3)
        linear.weight.copy_(Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        # bias shape: (out_features,) = (2,)
        linear.bias.copy_(Tensor([0.1, 0.2]))
    
    # Test input
    input_tensor = Tensor([[1.0, 2.0, 3.0]])  # shape: (1, 3)
    
    # Forward pass
    output = linear(input_tensor)
    
    # Expected output: input @ weight.T + bias
    # [1, 2, 3] @ [[1, 4], [2, 5], [3, 6]] + [0.1, 0.2]
    # = [1*1 + 2*2 + 3*3, 1*4 + 2*5 + 3*6] + [0.1, 0.2]
    # = [14, 32] + [0.1, 0.2] = [14.1, 32.2]
    expected_output = Tensor([[14.1, 32.2]])
    
    assert tensor_close(output, expected_output), \
        f"Expected {expected_output.tolist()}, got {output.tolist()}"
    
    return True

@testcase(name="linear_module_properties", score=10)
def test_linear_module_properties():
    """Test Linear module properties"""
    # Test with bias
    linear_with_bias = Linear(4, 3, bias=True)
    
    # Test it's a Module
    assert isinstance(linear_with_bias, Module), "Linear should inherit from Module"
    
    # Test parameters
    params = list(linear_with_bias.parameters())
    assert len(params) == 2, f"Linear with bias should have 2 parameters, got {len(params)}"
    
    # Check parameter shapes
    weight = linear_with_bias.weight
    bias = linear_with_bias.bias
    assert tuple(weight.shape) == (3, 4), f"Weight shape should be (3, 4), got {weight.shape}"
    assert tuple(bias.shape) == (3,), f"Bias shape should be (3,), got {bias.shape}"
    
    # Test without bias
    linear_no_bias = Linear(4, 3, bias=False)
    params_no_bias = list(linear_no_bias.parameters())
    # Should only have weight parameter when bias=False
    non_none_params = [p for p in params_no_bias if p is not None]
    assert len(non_none_params) == 1, f"Linear without bias should have 1 non-None parameter, got {len(non_none_params)}"
    assert linear_no_bias.bias is None, "Linear without bias should have bias=None"
    
    return True

@testcase(name="linear_batch_processing", score=10)
def test_linear_batch_processing():
    """Test Linear layer with batch processing"""
    linear = Linear(4, 2, bias=True)
    
    # Set known weights for predictable output
    with no_grad():
        linear.weight.copy_(Tensor([[1.0, 0.5, -1.0, 0.0], [0.0, 1.0, 0.5, -0.5]]))
        linear.bias.copy_(Tensor([0.1, -0.1]))
    
    # Test with batch input
    batch_size = 3
    input_tensor = Tensor([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 1.0, 0.0, -1.0],
        [0.5, 0.5, 0.5, 0.5]
    ])
    
    output = linear(input_tensor)
    
    # Check output shape
    expected_shape = (batch_size, 2)
    assert tuple(output.shape) == expected_shape, \
        f"Expected output shape {expected_shape}, got {output.shape}"
    
    # Verify first sample calculation manually
    # First row: [1, 2, 3, 4] @ [[1, 0], [0.5, 1], [-1, 0.5], [0, -0.5]] + [0.1, -0.1]
    # = [1*1 + 2*0.5 + 3*(-1) + 4*0, 1*0 + 2*1 + 3*0.5 + 4*(-0.5)] + [0.1, -0.1]
    # = [1 + 1 - 3 + 0, 0 + 2 + 1.5 - 2] + [0.1, -0.1]
    # = [-1, 1.5] + [0.1, -0.1] = [-0.9, 1.4]
    expected_first_row = Tensor([[-0.9, 1.4]])
    assert tensor_close(output[0:1], expected_first_row, rtol=1e-6), \
        f"First row calculation failed. Expected {expected_first_row.tolist()}, got {output[0:1].tolist()}"
    
    return True

@testcase(name="linear_repr", score=10)
def test_linear_repr():
    """Test string representation of Linear module"""
    linear1 = Linear(784, 128, bias=True)
    linear2 = Linear(10, 5, bias=False)
    
    # Test repr contains essential information
    repr1 = str(linear1)
    repr2 = str(linear2)
    
    assert "Linear" in repr1, f"Linear repr should contain 'Linear': {repr1}"
    assert "Linear" in repr2, f"Linear repr should contain 'Linear': {repr2}"
    
    # The repr should indicate the layer configuration
    # Based on tutorial, it should show in_features, out_features, bias
    print(f"Linear reprs:")
    print(f"  With bias: {repr1}")
    print(f"  Without bias: {repr2}")
    
    return True

@testcase(name="linear_tanh_combination", score=10)
def test_linear_tanh_combination():
    """Test combining Linear and Tanh modules"""
    # Create a simple two-layer network: Linear -> Tanh -> Linear
    linear1 = Linear(2, 3, bias=True)  
    tanh = Tanh()
    linear2 = Linear(3, 1, bias=True)
    
    # Set known weights
    with no_grad():
        # First layer: 2 -> 3
        linear1.weight.copy_(Tensor([[1.0, 0.5], [2.0, -1.0], [0.0, 1.5]]))
        linear1.bias.copy_(Tensor([0.1, -0.2, 0.3]))
        
        # Second layer: 3 -> 1  
        linear2.weight.copy_(Tensor([[0.5, 1.0, -0.5]]))
        linear2.bias.copy_(Tensor([0.2]))
    
    # Test input
    input_tensor = Tensor([[2.0, 3.0]])
    
    # Forward pass through the network
    linear1_output = linear1(input_tensor)  # Apply first linear layer
    tanh_output = tanh(linear1_output)      # Apply tanh activation  
    output = linear2(tanh_output)           # Apply second linear layer
    
    # Verify intermediate results
    # First linear: same as previous tests -> [3.6, 0.8, 4.8]
    expected_linear1_output = Tensor([[3.6, 0.8, 4.8]])
    assert tensor_close(linear1_output, expected_linear1_output, rtol=1e-4), \
        f"First linear layer failed. Expected {expected_linear1_output.tolist()}, got {linear1_output.tolist()}"
    
    # After tanh: tanh([3.6, 0.8, 4.8])
    # tanh is approximately [0.999, 0.664, 0.999] for these values
    expected_tanh_output = tanh(expected_linear1_output)
    assert tensor_close(tanh_output, expected_tanh_output, rtol=1e-4), \
        f"Tanh output failed. Expected {expected_tanh_output.tolist()}, got {tanh_output.tolist()}"
    
    # Final output shape should be correct
    assert tuple(output.shape) == (1, 1), f"Final output shape should be (1, 1), got {output.shape}"
    
    # The exact value depends on tanh, but it should be a reasonable number
    output_val = output.item()
    assert -10 < output_val < 10, f"Output value {output_val} seems unreasonable"
    
    return True

if __name__ == "__main__":
    set_debug_mode(True)
    
    print("Testing Part 2: Simplest Concrete Modules")
    print("=" * 50)
    
    # Run all tests
    test_functions = [
        test_tanh_forward,
        test_tanh_module_properties,
        test_tanh_batch_processing,
        test_tanh_repr,
        test_linear_forward_deterministic,
        test_linear_module_properties,
        test_linear_batch_processing,
        test_linear_repr,
        test_linear_tanh_combination,
    ]
    
    for test_func in test_functions:
        try:
            print(f"\nRunning {test_func.__name__}...")
            result = test_func()
            if result:
                print(f"✓ {test_func.__name__} passed")
            else:
                print(f"✗ {test_func.__name__} failed")
        except Exception as e:
            print(f"✗ {test_func.__name__} failed with error: {e}")
    
    print("\n" + "=" * 50)
    grader_summary()
