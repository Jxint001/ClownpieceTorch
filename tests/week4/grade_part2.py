from graderlib import testcase, grader_summary, value_close
import clownpiece as CP
from clownpiece.utils.optim.optimizer import SGD, Adam
from clownpiece.nn.module import Parameter

@testcase("SGD with momentum and weight decay", 10)
def test_sgd_momentum_weight_decay():
    p = Parameter(CP.Tensor([1.0, 2.0]))
    p.grad = CP.Tensor([0.1, 0.2])
    opt = SGD([p], lr=0.5, momentum=0.9, weight_decay=0.1)
    opt.step()
    # Should apply both grad and weight decay, and use momentum
    # Just check value changed and not equal to vanilla SGD
    assert not value_close(p, CP.Tensor([0.95, 1.9]))

@testcase("Adam bias correction", 10)
def test_adam_bias_correction():
    p = Parameter(CP.Tensor([1.0, 2.0]))
    p.grad = CP.Tensor([0.1, 0.2])
    opt = Adam([p], lr=0.1, betas=(0.9, 0.999))
    for _ in range(10):
        opt.step()
    # After several steps, Adam should have applied bias correction
    assert not value_close(p, CP.Tensor([1.0, 2.0]))

@testcase("SGD momentum accumulation", 10)
def test_sgd_momentum_accumulation():
    p = Parameter(CP.Tensor([1.0, 2.0]))
    p.grad = CP.Tensor([0.1, 0.2])
    opt = SGD([p], lr=0.5, momentum=0.9)
    
    # First step - no momentum yet
    initial_value = p.clone()
    opt.step()
    first_step_value = p.clone()
    
    # Second step with same gradient - should have momentum effect
    p.grad = CP.Tensor([0.1, 0.2])
    opt.step()
    second_step_value = p.clone()
    
    # The second step should move further due to momentum
    first_change = initial_value - first_step_value
    second_change = first_step_value - second_step_value
    # Second change should be larger due to momentum
    assert second_change.abs().sum() > first_change.abs().sum()

@testcase("Adam with weight decay", 10)
def test_adam_weight_decay():
    # Create two separate Parameter objects
    p_no_decay = Parameter(CP.Tensor([1.0, 2.0]))
    p_decay = Parameter(CP.Tensor([1.0, 2.0]))
    
    # Adam without weight decay
    opt_no_decay = Adam([p_no_decay], lr=0.1, weight_decay=0.0)
    p_no_decay.grad = CP.Tensor([0.1, 0.2])
    opt_no_decay.step()
    
    # Adam with weight decay
    opt_decay = Adam([p_decay], lr=0.1, weight_decay=0.1)
    p_decay.grad = CP.Tensor([0.1, 0.2])
    opt_decay.step()
    
    # With weight decay, parameters should be smaller (closer to zero)
    assert p_decay.abs().sum() < p_no_decay.abs().sum()

@testcase("Adam epsilon parameter", 10)
def test_adam_epsilon():
    # Create two separate Parameter objects
    p_small_eps = Parameter(CP.Tensor([1.0, 2.0]))
    p_large_eps = Parameter(CP.Tensor([1.0, 2.0]))
    
    # With small epsilon
    opt_small_eps = Adam([p_small_eps], lr=0.1, eps=1e-8)
    p_small_eps.grad = CP.Tensor([0.1, 0.2])
    opt_small_eps.step()
    
    # With larger epsilon
    opt_large_eps = Adam([p_large_eps], lr=0.1, eps=1e-4)
    p_large_eps.grad = CP.Tensor([0.1, 0.2])
    opt_large_eps.step()
    
    # Both should change the parameters
    assert not value_close(p_large_eps, CP.Tensor([1.0, 2.0]))
    assert not value_close(p_small_eps, CP.Tensor([1.0, 2.0]))

if __name__ == "__main__":
    test_sgd_momentum_weight_decay()
    test_adam_bias_correction()
    test_sgd_momentum_accumulation()
    test_adam_weight_decay()
    test_adam_epsilon()
    grader_summary()
