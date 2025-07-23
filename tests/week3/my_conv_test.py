from graderlib import set_debug_mode, testcase, grader_summary, value_close
import clownpiece as CP
from clownpiece.nn.layers import Conv2d

@testcase("Conv2d simple 1x1 kernel, no bias", 10)
def test_conv2d_1x1():
    # 输入: batch=1, in_channels=1, H=3, W=3
    x = CP.Tensor([[[[1,2,3],[4,5,6],[7,8,9]]]])  # shape (1,1,3,3)
    conv = Conv2d(in_channels=1, out_channels=1, kernel_height=1, kernel_width=1)
    conv.weight = CP.Tensor([[[[2]]]])  # shape (1,1,1,1)
    conv.bias = CP.Tensor([0])
    y = conv(x)
    # 期望输出: 每个元素都乘以2
    expected = CP.Tensor([[[[2,4,6],[8,10,12],[14,16,18]]]])
    assert value_close(y, expected)

@testcase("Conv2d 3x3 kernel, padding=same", 10)
def test_conv2d_3x3_same():
    x = CP.Tensor([[[[1,2,3],[4,5,6],[7,8,9]]]])  # shape (1,1,3,3)
    conv = Conv2d(in_channels=1, out_channels=1, kernel_height=3, kernel_width=3)
    # kernel全部为1
    conv.weight = CP.Tensor([[[[1,1,1],[1,1,1],[1,1,1]]]])
    conv.bias = CP.Tensor([0])
    y = conv(x)
    # 手动卷积结果
    expected = CP.Tensor([[[[12,21,16],[27,45,33],[24,39,28]]]])
    assert value_close(y, expected)

@testcase("Conv2d multi-channel", 10)
def test_conv2d_multi_channel():
    x = CP.Tensor([[
        [[1,2],[3,4]],    # channel 1
        [[5,6],[7,8]]     # channel 2
    ]])  # shape (1,2,2,2)
    conv = Conv2d(in_channels=2, out_channels=1, kernel_height=2, kernel_width=2)
    # kernel: 对两个通道分别乘以1和2
    conv.weight = CP.Tensor([[
        [[1,1],[1,1]],    # channel 1
        [[2,2],[2,2]]     # channel 2
    ]])  # shape (1,2,2,2)
    conv.bias = CP.Tensor([0])
    y = conv(x)
    # 期望输出: sum(输入*kernel) = (1+2+3+4)*1 + (5+6+7+8)*2 = 10*1 + 26*2 = 10+52=62
    expected = CP.Tensor([[[[62, 34],
                [37, 20]]]])
    assert value_close(y, expected)

@testcase("Conv2d backward simple", 10)
def test_conv2d_backward():
    x = CP.Tensor([[[[1., 2.], [3., 4.]]]], requires_grad=True)  # shape (1,1,2,2)
    conv = Conv2d(in_channels=1, out_channels=1, kernel_height=2, kernel_width=2)
    conv.weight = CP.Tensor([[[[1., 1.], [1., 1.]]]])  # shape (1,1,2,2)
    conv.bias = CP.Tensor([0.])
    y = conv(x)
    loss = y.sum()
    loss.backward()
    # 期望梯度：每个输入元素都被卷积核覆盖了4次
    expected_grad = CP.Tensor([[[[1, 2],
                                [2, 4]]]])
    assert value_close(x.grad, expected_grad)

if __name__ == "__main__":
    set_debug_mode(True)
    test_conv2d_1x1()
    test_conv2d_3x3_same()
    test_conv2d_multi_channel()
    test_conv2d_backward()
    grader_summary()