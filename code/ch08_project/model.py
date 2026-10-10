"""第 8 章工程版：模型定义。

从第 3 章搬来 `SimpleCNN`，**去掉教学用的 `init_mode` 分支**。

为什么去掉？第 3 章那个 `init_mode` 参数是为了演示"初始化方差有多重要"
（zeros 学不动、large 训不起来）。演示完了，结论是"kaiming 就行"。
工程代码里留着一个只需要传 "kaiming" 的参数，等于给每个读代码的人
增加一次"这个参数我该传什么"的思考 —— 而正确答案永远只有一个。

**教学的版本要保留所有岔路，工程的版本只保留结论。** 这是同一个逻辑
在两个场景下的写法差异，本章 8.4 节专门对比这一类差异。
"""
import torch.nn as nn


class SimpleCNN(nn.Module):
    """两层卷积 + 两层全连接的 CNN（第 3 章那个，参数 225,034）。

    结构：
        输入 1x28x28
        -> Conv2d(1 -> 32, 3x3)  -> ReLU -> MaxPool2d(2)    [28x28 -> 14x14]
        -> Conv2d(32 -> 64, 3x3) -> ReLU -> MaxPool2d(2)    [14x14 -> 5x5]
        -> Flatten               [64x5x5 = 1600]
        -> Linear(1600 -> 128) -> ReLU
        -> Linear(128 -> 10)     输出 logits，不接 softmax（交给 CrossEntropyLoss）
    """

    def __init__(self) -> None:
        super().__init__()

        # --- 特征提取：卷积层"看图" ---
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # --- 分类：全连接层"下结论" ---
        # 1600 是按真实前向打印出的形状填的（第 3 章的 trace_shapes），不是手算的
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1600, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

        self._init_weights()

    def _init_weights(self) -> None:
        """He 初始化（kaiming）。

        第 3 章的实测结论：默认初始化能用，kaiming 略好，zeros 完全学不动。
        所以这里直接写死 kaiming，不再留开关。
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # mode="fan_out"    方差按这一层"输出神经元个数"缩放
                # nonlinearity="relu" 补偿 ReLU 砍掉一半负值造成的方差损失
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # 偏置不需要随机，置 0
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)


def build_model(cfg) -> nn.Module:
    """按配置建模型。

    现在 `cfg` 没被用到（模型结构写死在 SimpleCNN 里）。**这不是多余的参数** ——
    它把"模型由配置决定"这个接口先固定下来。等以后要加 `cfg.hidden_dim`
    或者按 `cfg.arch` 选不同结构时，改这个函数就行，
    调用方 `train.py` 一个字都不用动。

    这是"接口先行"的一个小例子：**先让依赖方向正确，再往里填内容。**
    """
    return SimpleCNN()


def count_params(model: nn.Module) -> int:
    """可训练参数总量。第 3 章用这个数对比 CNN 和 MLP（225,034 vs 269,322）。"""
    return sum(p.numel() for p in model.parameters())
