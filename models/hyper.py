import torch
import torch.nn as nn
import torch.nn.functional as F


class hyperConv(nn.Module):
    def __init__(
            self,
            style_dim,
            dim_in,
            dim_out,
            ksize,
            stride=1,
            padding=None,
            bias=True,
            weight_dim=8,
            ndims=2,
    ):
        super().__init__()
        assert ndims in [2, 3]
        self.ndims = ndims
        self.dim_out = dim_out
        self.stride = stride
        self.bias = bias
        self.weight_dim = weight_dim
        self.fc = nn.Linear(style_dim, weight_dim)
        self.kshape = [dim_out, dim_in, ksize, ksize] if self.ndims == 2 else [dim_out, dim_in, ksize, ksize, ksize]
        self.padding = (ksize - 1) // 2 if padding is None else padding

        self.param = nn.Parameter(torch.randn(*self.kshape, weight_dim).type(torch.float32))
        nn.init.kaiming_normal_(self.param, a=0, mode='fan_in')

        if self.bias is True:
            self.fc_bias = nn.Linear(style_dim, weight_dim)
            self.b = nn.Parameter(torch.randn(self.dim_out, weight_dim).type(torch.float32))
            nn.init.constant_(self.b, 0.0)

        self.conv = getattr(F, 'conv%dd' % self.ndims)  # 如果 'self.ndims' 的值是 2，那么 'conv%dd' % self.ndims 就会变成 'conv2d'

    def forward(self, x, s):    # x: input feature maps; s: target sequence code;
        # print("x.shape: ", x.shape)     # [4, 256, 66, 66]
        # print("s.shape: ", s.shape)     # [4, 20]
        # print("fc(s).shape: ", self.fc(s).shape)    # [4, 8]
        # print("weight dim: ", self.weight_dim)      # 8
        # print("param.shape: ", self.param.shape)    # [256, 256, 3, 3, 8]
        # print("bias: ", self.bias)
        # print(self.fc(s).view(self.weight_dim, 1).shape)
        kernel = torch.matmul(self.param, self.fc(s).view(self.weight_dim, 1)).view(*self.kshape)
        # print('++++++++++++++++++++++++++++++++++++++')
        if self.bias is True:
            bias = torch.matmul(self.b, self.fc_bias(s).view(self.weight_dim, 1)).view(self.dim_out)
            return self.conv(x, weight=kernel, bias=bias, stride=self.stride, padding=self.padding)
        else:
            return self.conv(x, weight=kernel, stride=self.stride, padding=self.padding)


class hyperResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, style_dim, dim, padding_type, norm_layer, use_bias, weight_dim=8, ndims=2):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(hyperResnetBlock, self).__init__()
        self.ndims = ndims
        ReflectionPad = getattr(nn, 'ReflectionPad%dd' % self.ndims)
        ReplicationPad = getattr(nn, 'ReplicationPad%dd' % self.ndims)

        p = 0
        if padding_type == 'reflect':
            self.pad1 = ReflectionPad(1)
        elif padding_type == 'replicate':
            self.pad1 = ReplicationPad(1)
        elif padding_type == 'zero':
            self.pad1 = nn.Identity()
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        self.conv1 = hyperConv(style_dim, dim, dim, ksize=3, padding=p, bias=use_bias, weight_dim=weight_dim,
                               ndims=self.ndims)
        if norm_layer is not None:
            self.norm1 = nn.Sequential(norm_layer(dim), nn.LeakyReLU(0.2, True))
        else:
            self.norm1 = nn.LeakyReLU(0.2, True)

        p = 0
        if padding_type == 'reflect':
            self.pad2 = ReflectionPad(1)
        elif padding_type == 'replicate':
            self.pad2 = ReplicationPad(1)
        elif padding_type == 'zero':
            self.pad2 = nn.Identity()
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        self.conv2 = hyperConv(style_dim, dim, dim, ksize=3, padding=p, bias=use_bias, weight_dim=weight_dim,
                               ndims=self.ndims)
        if norm_layer is not None:
            self.norm2 = norm_layer(dim)
        else:
            self.norm2 = nn.Identity()

    def forward(self, x, s):
        """Forward function (with skip connections)"""
        y = self.norm1(self.conv1(self.pad1(x), s))     #######
        y = self.norm2(self.conv2(self.pad2(y), s))
        # 
        # print("--------------------")
        # print(x.shape, y.shape)
        # print("--------------------")
        out = x + y  # add skip connections
        return out


class hyperDecoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.ndims = args['seq2seq']['ndims']
        self.c_dec = args['seq2seq']['c_dec']
        self.k_dec = args['seq2seq']['k_dec']
        self.s_dec = args['seq2seq']['s_dec']
        self.nres_dec = args['seq2seq']['nres_dec']
        self.style_dim = args['seq2seq']['c_s']
        self.weight_dim = args['seq2seq']['c_w']

        norm = args['seq2seq']['norm']
        self.norm = getattr(nn, '%s%dd' % (norm, self.ndims)) if norm is not None else None
        ReflectionPad = getattr(nn, 'ReflectionPad%dd' % self.ndims)

        c_pre = args['seq2seq']['c_enc'][-1]
        self.res = nn.ModuleList()
        for _ in range(self.nres_dec):
            self.res.append(
                hyperResnetBlock(self.style_dim, c_pre, padding_type='reflect', norm_layer=self.norm, use_bias=True,
                                 weight_dim=self.weight_dim, ndims=self.ndims))

        self.up = nn.Upsample(scale_factor=2, mode='bilinear' if self.ndims == 2 else 'trilinear')

        self.pads = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for c, k, s in zip(self.c_dec[:-1], self.k_dec[:-1], self.s_dec[:-1]):
            self.pads.append(ReflectionPad((k - 1) // 2))
            self.convs.append(
                hyperConv(self.style_dim, c_pre, c, ksize=k, stride=s, padding=0, weight_dim=self.weight_dim,
                          ndims=self.ndims))
            if self.norm is not None:
                self.norms.append(nn.Sequential(self.norm(c), nn.LeakyReLU(0.2, True)))
            else:
                self.norms.append(nn.LeakyReLU(0.2, True))
            c_pre = c

        self.pad_last = ReflectionPad((self.k_dec[-1] - 1) // 2)
        self.conv_last = hyperConv(self.style_dim, dim_in=c_pre, dim_out=self.c_dec[-1], ksize=self.k_dec[-1],
                                   padding=0, weight_dim=self.weight_dim, ndims=self.ndims)
        self.act_last = nn.Tanh()

    def forward(self, x, s):
        for res in self.res:
            x = res(x, s)

        for pad, conv, norm in zip(self.pads, self.convs, self.norms):
            x = self.up(x)
            x = norm(conv(pad(x), s))

        # print(f"hyper decoder input x: {x.shape}")

        x = self.act_last(self.conv_last(self.pad_last(x), s))

        # print(f"hyper decoder output x: {x.shape}")

        return x
