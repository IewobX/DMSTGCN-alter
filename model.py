import torch
import torch.nn as nn
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nwv->ncwl', (x, A))
        return x.contiguous()  # 把tensor变成在内存中连续分布的形式
                               # torch.view等方法操作需要连续的Tensor；连续的Tensor，语义上相邻的元素，在内存中也是连续的，语义和内存顺序的一致性是缓存友好的，以提升CPU获取操作数据的速度

# 在某个数据上应用一个线性转换，公式表达就是y=xA^T+b
# bias: 默认为True.如果设置成false，则这个线性层不会加上bias。值从均匀分布U(-\sqrt{k},\sqrt{k})中获取
class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        # print("x.shape", x.shape) x.shape torch.Size([64, 32, 307, 7])
        for a in support:
            x1 = self.nconv(x, a)
            # print("x1.shape", x1.shape)  x1.shape torch.Size([64, 32, 307, 7])
            out.append(x1) #把新结果x1添加到out中
            for k in range(2, self.order + 1): #for 2 in (2,3)...
                # print("k", k) k=2
                x2 = self.nconv(x1, a)
                # print("x2.shape", x2.shape) x2.shape torch.Size([64, 32, 307, 7])
                out.append(x2)
                x1 = x2  #把x2的值赋值给x1

        h = torch.cat(out, dim=1)  #把out值按列拼接
        # print("h.shape", h.shape)  h.shape torch.Size([64, 96, 307, 7])
        h = self.mlp(h)
        # print("h.shape", h.shape)  h.shape torch.Size([64, 32, 307, 7])
        h = F.dropout(h, self.dropout, training=self.training)
        # print("h.shape", h.shape)   h.shape torch.Size([64, 32, 307, 7])
        return h

class dilated_1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,2,2,2]
        self.tconv = nn.Conv2d(cin,cout,(1,2),dilation=(1,dilation_factor))

    def forward(self,input):
        x = self.tconv(input)
        return x
class DMSTGCN(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3,
                 out_dim=12, residual_channels=16, dilation_channels=16, end_channels=512,
                 kernel_size=2, blocks=4, layers=2, days=288, dims=40, order=2, in_dim=9, normalization="batch"):
        super(DMSTGCN, self).__init__()
        skip_channels = 8
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.normal = nn.ModuleList()
        self.gconv = nn.ModuleList()

        # self.filter_convs_a = nn.ModuleList()
        # self.gate_convs_a = nn.ModuleList()
        # self.residual_convs_a = nn.ModuleList()
        # self.skip_convs_a = nn.ModuleList()
        # self.normal_a = nn.ModuleList()
        # self.gconv_a = nn.ModuleList()
        #
        # self.gconv_a2p = nn.ModuleList()
        #
        # self.start_conv_a = nn.Conv2d(in_channels=in_dim,
        #                               out_channels=residual_channels,
        #                               kernel_size=(1, 1))

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        receptive_field = 1

        self.supports_len = 1
        self.nodevec_p1 = nn.Parameter(torch.randn(days, dims).to(device), requires_grad=True).to(device) #requires_grad自动
        self.nodevec_p2 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec_p3 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec_pk = nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device)
        # self.nodevec_a1 = nn.Parameter(torch.randn(days, dims).to(device), requires_grad=True).to(device)
        # self.nodevec_a2 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        # self.nodevec_a3 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        # self.nodevec_ak = nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device)
        # self.nodevec_a2p1 = nn.Parameter(torch.randn(days, dims).to(device), requires_grad=True).to(device)
        # self.nodevec_a2p2 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        # self.nodevec_a2p3 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        # self.nodevec_a2pk = nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions  （改动部分：将扩张卷积换成扩张inception）
                # self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                #                                    out_channels=dilation_channels,
                #                                    kernel_size=(1, kernel_size), dilation=new_dilation))
                #
                # self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                #                                  out_channels=dilation_channels,
                #                                  kernel_size=(1, kernel_size), dilation=new_dilation))
                self.filter_convs.append(
                    dilated_inception(residual_channels, dilation_channels, dilation_factor=new_dilation))

                self.gate_convs.append(
                    dilated_inception(residual_channels, dilation_channels, dilation_factor=new_dilation))

                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))

                # self.filter_convs_a.append(nn.Conv2d(in_channels=residual_channels,
                #                                      out_channels=dilation_channels,
                #                                      kernel_size=(1, kernel_size), dilation=new_dilation))
                #
                # self.gate_convs_a.append(nn.Conv1d(in_channels=residual_channels,
                #                                    out_channels=dilation_channels,
                #                                    kernel_size=(1, kernel_size), dilation=new_dilation))

                # self.filter_convs_a.append(
                #     dilated_inception(residual_channels, dilation_channels, dilation_factor=new_dilation))
                #
                # self.gate_convs_a.append(
                #     dilated_inception(residual_channels, dilation_channels, dilation_factor=new_dilation))

                # 1x1 convolution for residual connection
                # self.residual_convs_a.append(nn.Conv1d(in_channels=dilation_channels,
                #                                        out_channels=residual_channels,
                #                                        kernel_size=(1, 1)))
                if normalization == "batch":
                    self.normal.append(nn.BatchNorm2d(residual_channels))
                    # self.normal_a.append(nn.BatchNorm2d(residual_channels))
                elif normalization == "layer":
                    self.normal.append(nn.LayerNorm([residual_channels, num_nodes, 13 - receptive_field - new_dilation + 1]))
                    # self.normal_a.append(nn.LayerNorm([residual_channels, num_nodes, 13 - receptive_field - new_dilation + 1]))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                self.gconv.append(
                    gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))
                # self.gconv_a.append(
                #     gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))
                # self.gconv_a2p.append(
                #     gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))

        self.relu = nn.ReLU(inplace=True)

        self.end_conv_1 = nn.Conv2d(in_channels=304,
                                    out_channels=304,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=304,
                                    out_channels=12,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def dgconstruct(self, time_embedding, source_embedding, target_embedding, core_embedding):
        adp = torch.einsum('ai, ijk->ajk', time_embedding, core_embedding)
        # print("adp.shape", adp.shape)  adp.shape torch.Size([64, 32, 32])
        adp = torch.einsum('bj, ajk->abk', source_embedding, adp)
        # print("adp.shape", adp.shape)  adp.shape torch.Size([64, 307, 32])
        adp = torch.einsum('ck, abk->abc', target_embedding, adp)
        # print("adp.shape", adp.shape)  adp.shape torch.Size([64, 307, 307])
        adp = F.softmax(F.relu(adp), dim=2)
        # print("adp.shape", adp.shape)  adp.shape torch.Size([64, 307, 307])
        return adp

    def forward(self, inputs, ind):
        """
        input: (B, F, N, T)
        """
        in_len = inputs.size(3)  #返回第四维的元素个数 填充部分，如果输入长度小于感受野，填充满
        # print("in_len", in_len) in_len=13
        if in_len < self.receptive_field:
            xo = nn.functional.pad(inputs, (self.receptive_field - in_len, 0, 0, 0)) #填充区域
        else:
            xo = inputs #输入大于感受区域，把输入赋值给xo
            # print("xo.shape", xo.shape) xo.shape torch.Size([64, 2, 307, 13])
        x = self.start_conv(xo[:, [0]]) #取xo的第一列，经过start_conv得到x
        # print("x.shape", x.shape) x.shape torch.Size([64, 32, 307, 13])
        # x_a = self.start_conv_a(xo[:, [1]])
        skip = 0

        # dynamic graph construction
        adp = self.dgconstruct(self.nodevec_p1[ind], self.nodevec_p2, self.nodevec_p3, self.nodevec_pk)
        # adp_a = self.dgconstruct(self.nodevec_a1[ind], self.nodevec_a2, self.nodevec_a3, self.nodevec_ak)
        # adp_a2p = self.dgconstruct(self.nodevec_a2p1[ind], self.nodevec_a2p2, self.nodevec_a2p3, self.nodevec_a2pk)

        new_supports = [adp]
        # new_supports_a = [adp_a]
        # new_supports_a2p = [adp_a2p]

        for i in range(self.layers * self.layers):
            # tcn for primary part
            residual = x
            # print("x.shape", x.shape) 0层：[64, 32, 307, 13]
            # print("residual.shape",residual.shape)
            # print("self.filter_convs[i]",self.filter_convs[i])
            filter = self.filter_convs[i](residual)
            # print("residual.shape", residual.shape)
            # print("self.filter_convs[i]", self.filter_convs[i])
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            # print("x.shape", x.shape) [64, 32, 307, 7]

            # tcn for auxiliary part (改动部分:删除辅助属性部分的tcn)
            # residual_a = x_a
            # filter_a = self.filter_convs_a[i](residual_a)
            # filter_a = torch.tanh(filter_a)
            # gate_a = self.gate_convs_a[i](residual_a)
            # gate_a = torch.sigmoid(gate_a)
            # x_a = filter_a * gate_a

            # skip connection
            s = x
            s = self.skip_convs[i](s)
            # print("s.shape", s.shape) ([64, 8, 307, 7])
            if isinstance(skip, int):  # 判断两个类型是否相同
                skip = s.transpose(2, 3).reshape([s.shape[0], -1, s.shape[2], 1]).contiguous()  #重塑输出的向量 # 输出张量第0维第2维 # 自动计算新的维数 # 保证Tensor是连续的
            else:
                skip = torch.cat([s.transpose(2, 3).reshape([s.shape[0], -1, s.shape[2], 1]), skip], dim=1).contiguous()
            # print("skip.shape", skip.shape) ([64, 56, 307, 1])
            # dynamic graph convolutions ([64, 32, 307, 7])
            x = self.gconv[i](x, new_supports)
            # x_a = self.gconv_a[i](x_a, new_supports_a)

            # multi-faceted fusion module  （改动部分：删除主辅融合部分）
            # x_a2p = self.gconv_a2p[i](x_a, new_supports_a2p)
            # x = x_a2p + x

            # residual and normalization ([64, 32, 307, 7])
            # x_a = x_a + residual_a[:, :, :, -x_a.size(3):]  # 倒数x_a.size(3)个数
            residual = x
            x = x + residual[:, :, :, -x.size(3):]
            x = self.normal[i](x)
            # x_a = self.normal_a[i](x_a)

        # output layer
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        # print("self.end_conv_1(x)",self.end_conv_1(x))
        x = self.end_conv_2(x)
        # print("self.end_conv_2(x)",self.end_conv_2(x))
        return x

class dilated_inception(nn.Module):   #dilated_inception是module的一个子类，
    def __init__(self, cin, cout, dilation_factor=2):  #init定义了这个类里面都用到了哪些东西
        super(dilated_inception, self).__init__()   #调用module函数这是每个函数既定的要求
        self.tconv = nn.ModuleList() #把nn.ModuleList里的东西放到tconv中
        self.kernel_set = [2, 2, 2, 2]  #把[2,3,6,7]放到kernel_set中
        cout = int(cout/len(self.kernel_set))  #把输出除以卷积核的长度后去整
        for kern in self.kernel_set:   #在kernel_set（2，2，2，2）中分别取值循环
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))  #self.tconv.append(nn.Conv2d(cin, cout, (1, 2), dilation=(1, 2))),将括号里的东西经过2d卷积后逐一添加到self.tconv中

    def forward(self, input):
        # print("input",input.shape)
        x = []
        for i in range(len(self.kernel_set)): #i循环
            x.append(self.tconv[i](input))
            # print("self.tconv[i]", self.tconv[i])
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        # print("x.shape", x.shape)
        return x
