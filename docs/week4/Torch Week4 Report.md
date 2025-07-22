# Torch Week4 Report

## Grade summary from grade_all.py
```
=== Final Summary ===
Total tests: 29 test cases, 290 points total
==================================================
Total Score: 290/290
Passed: 29
Failed: 0

Passed tests: CSVDataset basic, CSVDataset with transform, ImageDataset basic, ImageDataset with transform, Dataloader batching, Dataloader shuffle, Dataloader drop_last, Transform functions, SGD vanilla, Adam step, Optimizer add_param_group, Optimizer zero_grad, LambdaLR schedule, ExponentialLR schedule, StepLR schedule, LRScheduler get_last_lr, LRScheduler with last_epoch, SGD with momentum and weight decay, Adam bias correction, SGD momentum accumulation, Adam with weight decay, Adam epsilon parameter, LambdaLR custom lambda, ExponentialLR gamma=0.5, StepLR step_size=3, LambdaLR with multiple param groups, ExponentialLR different gamma values, StepLR with different step sizes, Scheduler step with explicit epoch
==================================================
```

## Challenges encountered and solutions
这周的内容确实比前一周工作量小。耗时较多的地方主要是 Optimizer。

一开始没明白参数具体要怎么控制 Optimizer 的 step ，把逻辑捋顺写第一版之后发现一直 AssertionError，好像 step 过程根本没修改任何应该被优化的张量。拷打 AI 发现是 python 中如果写 `a = a + 1` 其实是创建了新的 `a` ，而 step 中要用原地修改，所以要用 `copy_` 函数。（之前好像也有某处涉及到这一点但是我忘记了……）顺利通过测试点之后又发现 estate_value_predict 中 training loss 和 test loss 从头到尾基本没有变过，是一条平行于 x 轴的线，大为震惊，非常疑惑。最后查了半天发现是 Optimizer 的 __init__() 函数没考虑输入是一个生成器的情况…… 所以当时的程序确实没有在优化任何量……

终于写完啦！