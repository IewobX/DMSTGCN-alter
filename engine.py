import torch.optim as optim
from model import *
import util


class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid, dropout, normalization, lrate, wdecay, device, days=288,
                 dims=40, order=2):
        self.model = DMSTGCN(device, num_nodes, dropout, out_dim=seq_length, residual_channels=nhid,
                             dilation_channels=nhid, end_channels=nhid * 16, days=days, dims=dims, order=order,
                             in_dim=in_dim, normalization=normalization)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scaeduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=.3, patience=10, threshold=1e-3,
                                                              min_lr=1e-5, verbose=True)
        # 基于训练过程中的某些测量值对学习率进行动态的下降.
        # 当参考的评价指标停止改进时, 降低学习率, factor为每次下降的比例, 训练过程中, 当指标连续patience次数还没有改进时, 降低学习率;

        # 当网络的评价指标不在提升的时候，可以通过降低网络的学习率来提高网络性能
        # optimer指的是网络的优化器
        # mode(str) ，可选择‘min’或者‘max’，min表示当监控量停止下降的时候，学习率将减小，max表示当监控量停止上升的时候，学习率将减小。默认值为‘min’
        # factor学习率每次降低多少，new_lr = old_lr * factor
        # patience = 10，容忍网路的性能不提升的次数，高于这个次数就降低学习率
        # verbose（bool） - 如果为True，则为每次更新向stdout输出一条消息。 默认值：False
        # threshold（float） - 测量新最佳值的阈值，仅关注重大变化。 默认值：1e-4
        # cooldown： 减少lr后恢复正常操作之前要等待的时期数。 默认值：0。
        # min_lr, 学习率的下限
        # eps ，适用于lr的最小衰减。 如果新旧lr之间的差异小于eps，则忽略更新。 默认值：1e-8。
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5  # 用于梯度裁剪防止梯度爆炸

    def train(self, input, real_val, ind):
        self.model.train()
        self.optimizer.zero_grad()  # 梯度置零，也就是把loss关于weight的导数变成0
        input = nn.functional.pad(input, (1, 0, 0, 0))
        # print("input", input)
        output = self.model(input, ind)
        # print("output", output)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)  # 扩充维度
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        loss.backward()  # 反向传播求梯度
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)  # 控制W变化的不那么大，防止过拟合
        self.optimizer.step()  # 更新所有参数
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return mae, mape, rmse

    def eval(self, input, real_val, ind):
        self.model.eval()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input, ind)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return mae, mape, rmse
