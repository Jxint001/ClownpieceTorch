from graderlib import testcase, grader_summary, value_close
from clownpiece.utils.optim.lr_scheduler import LambdaLR, ExponentialLR, StepLR
from clownpiece.utils.optim.optimizer import Optimizer

class DummyOpt(Optimizer):
    def __init__(self):
        self.param_groups = [{'lr': 1.0}]
        self.defaults = {'lr': 1.0}
        self.state = {}
    def step(self): pass
    def zero_grad(self, set_to_None=True): pass

@testcase("LambdaLR custom lambda", 10)
def test_lambdalr_custom():
    opt = DummyOpt()
    sched = LambdaLR(opt, lr_lambda=lambda e: 1.0 if e < 2 else 0.1)
    sched.step() # epoch 0->1
    assert value_close(opt.param_groups[0]['lr'], 1.0)
    sched.step() # epoch 1->2
    assert value_close(opt.param_groups[0]['lr'], 0.1)
    sched.step() # epoch 2->3
    assert value_close(opt.param_groups[0]['lr'], 0.1)

@testcase("ExponentialLR gamma=0.5", 10)
def test_explr_half():
    opt = DummyOpt()
    sched = ExponentialLR(opt, gamma=0.5)
    sched.step()
    assert value_close(opt.param_groups[0]['lr'], 0.5)
    sched.step()
    assert value_close(opt.param_groups[0]['lr'], 0.25)

@testcase("StepLR step_size=3", 10)
def test_steplr_3():
    opt = DummyOpt()
    sched = StepLR(opt, step_size=3, gamma=0.1)
    sched.step() # epoch 0->1
    assert value_close(opt.param_groups[0]['lr'], 1.0)
    sched.step() # epoch 1->2
    assert value_close(opt.param_groups[0]['lr'], 1.0)
    sched.step() # epoch 2->3
    assert value_close(opt.param_groups[0]['lr'], 0.1)
    sched.step() # epoch 3->4
    assert value_close(opt.param_groups[0]['lr'], 0.1)
    sched.step() # epoch 4->5
    assert value_close(opt.param_groups[0]['lr'], 0.1)
    sched.step() # epoch 5->6
    assert value_close(opt.param_groups[0]['lr'], 0.01)

@testcase("LambdaLR with multiple param groups", 10)
def test_lambdalr_multiple_groups():
    class DummyOptMulti(Optimizer):
        def __init__(self):
            self.param_groups = [{'lr': 1.0}, {'lr': 2.0}]
            self.defaults = {'lr': 1.0}
            self.state = {}
        def step(self): pass
        def zero_grad(self, set_to_None=True): pass
    
    opt = DummyOptMulti()
    sched = LambdaLR(opt, lr_lambda=lambda e: 0.5**e)
    sched.step()
    assert value_close(opt.param_groups[0]['lr'], 0.5)
    assert value_close(opt.param_groups[1]['lr'], 1.0)  # 2.0 * 0.5
    sched.step()
    assert value_close(opt.param_groups[0]['lr'], 0.25)
    assert value_close(opt.param_groups[1]['lr'], 0.5)   # 2.0 * 0.25

@testcase("ExponentialLR different gamma values", 10)
def test_explr_different_gamma():
    opt = DummyOpt()
    
    # Test gamma close to 1 (slow decay)
    sched_slow = ExponentialLR(opt, gamma=0.95)
    sched_slow.step()
    assert value_close(opt.param_groups[0]['lr'], 0.95)
    
    # Reset optimizer
    opt.param_groups[0]['lr'] = 1.0
    
    # Test gamma close to 0 (fast decay)
    sched_fast = ExponentialLR(opt, gamma=0.1)
    sched_fast.step()
    assert value_close(opt.param_groups[0]['lr'], 0.1)

@testcase("StepLR with different step sizes", 10)
def test_steplr_different_steps():
    opt = DummyOpt()
    
    # Test step_size=1 (decay every epoch)
    sched = StepLR(opt, step_size=1, gamma=0.5)
    sched.step() # epoch 0->1
    assert value_close(opt.param_groups[0]['lr'], 0.5)
    sched.step() # epoch 1->2
    assert value_close(opt.param_groups[0]['lr'], 0.25)
    
    # Reset optimizer
    opt.param_groups[0]['lr'] = 1.0
    
    # Test step_size=5 (decay every 5 epochs)
    sched = StepLR(opt, step_size=5, gamma=0.1)
    for _ in range(4):
        sched.step()
        assert value_close(opt.param_groups[0]['lr'], 1.0)  # Should not change
    sched.step() # 5th step
    assert value_close(opt.param_groups[0]['lr'], 0.1)     # Should change now

@testcase("Scheduler step with explicit epoch", 10)
def test_scheduler_explicit_epoch():
    opt = DummyOpt()
    sched = ExponentialLR(opt, gamma=0.5)
    
    # Jump to epoch 3 directly
    sched.step(epoch=3)
    assert value_close(opt.param_groups[0]['lr'], 0.125)  # 0.5^3
    
    # Step normally from there
    sched.step()
    assert value_close(opt.param_groups[0]['lr'], 0.0625) # 0.5^4

if __name__ == "__main__":
    test_lambdalr_custom()
    test_explr_half()
    test_steplr_3()
    test_lambdalr_multiple_groups()
    test_explr_different_gamma()
    test_steplr_different_steps()
    test_scheduler_explicit_epoch()
    grader_summary()
