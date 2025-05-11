import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
import torch.nn.functional as F


def fuse_conv_and_bn(conv, bn):
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(
        conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def channel_shuffle(x, groups=2):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class RepDW(nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv(ed, ed, 5, 1, 2, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class ShuffleRepDwFusion(nn.Module):
    def __init__(self, c, s=False):
        super().__init__()
        # block
        self.lkrep = RepDW(c)
        self.fusion = Conv(c, c, 1)
        self.s = s

    def forward(self, x):
        x0, x1 = x.chunk(2, 1)
        x1 = self.fusion(self.lkrep(x1))
        x = torch.cat([x0, x1], 1)
        return channel_shuffle(x) if self.s else x


class GLKRep(nn.Module):
    def __init__(self, c1, c2, n=2):
        super().__init__()
        cm = int(c2 * 0.5)
        self.n = n
        # downs
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, 3, 2, g=c2, act=False)

        # block
        self.blocks = nn.Sequential(*[ShuffleRepDwFusion(cm // 2, s=(i != n-1)) for i in range(n)])

    def forward(self, x):
        x = self.cv2(self.cv1(x))
        x0, x1 = x.chunk(2, 1)
        x1 = self.blocks(x1)
        x = torch.cat([x0, x1], 1)
        return x


class FieldSelect(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim//2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn


class Unify(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [64, 128, 256]
        cm = int(ch[0] * 3)
        self.pool = nn.AvgPool2d(2, 2)
        self.align = nn.ModuleList([nn.Conv2d(c, ch[0], 1) for c in ch])
        self.fs = FieldSelect(cm)
        self.cve = Conv(cm, cm, 1)

    def forward(self, x):
        x[0] = self.align[0](self.pool(x[0]))
        x[1] = self.align[1](x[1])
        x[2] = self.align[2](F.interpolate(x[2], x[0].shape[2:], mode='nearest'))
        x = torch.cat(x, 1)
        x = self.fs(x)
        return self.cve(x)


class Resize(nn.Module):
    def __init__(self, s=1.5):
        super().__init__()
        self.s = s

    def forward(self, x):
        h, w = x.shape[2:]
        th, tw = int(h * self.s), int(w * self.s)
        x = F.interpolate(x, (th, tw), mode='nearest')
        return x
