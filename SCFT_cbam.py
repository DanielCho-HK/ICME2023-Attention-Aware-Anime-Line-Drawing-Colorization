import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import math


class ChannelAttention(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // ratio, False),
            nn.ReLU(),
            nn.Linear(in_channel // ratio, in_channel, False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        max_pool_out = self.max_pool(x).view(b, c)
        avg_pool_out = self.avg_pool(x).view(b, c)

        max_fc_out = self.fc(max_pool_out)
        avg_fc_out = self.fc(avg_pool_out)

        out = max_fc_out + avg_fc_out
        out = self.sigmoid(out).view(b, c, 1, 1)

        return x * out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_pool_out = torch.mean(x, dim=1, keepdim=True)
        pool_out = torch.cat([max_pool_out, mean_pool_out], dim=1)

        out = self.conv(pool_out)
        out = self.sigmoid(out)

        return x * out


class CBAM(nn.Module):
    def __init__(self, in_channel, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channel, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()

        def CL2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            layers += [CBAM(out_channels)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.LeakyReLU(0.2, inplace=False)]
            cbr = nn.Sequential(*layers)

            return cbr

        # conv_layer
        self.conv_1 = CL2d(in_channels, 16)           # 256
        self.conv_2 = CL2d(16, 16)                    # 256
        self.conv_3 = CL2d(16, 32, stride=2)          # 128
        self.conv_4 = CL2d(32, 32)                    # 128
        self.conv_5 = CL2d(32, 64, stride=2)          # 64
        self.conv_6 = CL2d(64, 64)                    # 64
        self.conv_7 = CL2d(64, 128, stride=2)         # 32
        self.conv_8 = CL2d(128, 128)                  # 32
        self.conv_9 = CL2d(128, 256, stride=2)        # 16
        self.conv_10 = CL2d(256, 256)                 # 16

        # down_sample_layer
        self.down_sampling = nn.AdaptiveAvgPool2d((16, 16))


    def forward(self, x):
        f1 = self.conv_1(x)
        f2 = self.conv_2(f1)
        f3 = self.conv_3(f2)
        f4 = self.conv_4(f3)
        f5 = self.conv_5(f4)
        f6 = self.conv_6(f5)
        f7 = self.conv_7(f6)
        f8 = self.conv_8(f7)
        f9 = self.conv_9(f8)
        f10 = self.conv_10(f9)

        F = [f9, f8, f7, f6, f5, f4, f3, f2, f1]

        v1 = self.down_sampling(f1)
        v2 = self.down_sampling(f2)
        v3 = self.down_sampling(f3)
        v4 = self.down_sampling(f4)
        v5 = self.down_sampling(f5)
        v6 = self.down_sampling(f6)
        v7 = self.down_sampling(f7)
        v8 = self.down_sampling(f8)

        V = torch.cat([v1, v2, v3, v4, v5, v6, v7, v8, f9, f10], dim=1)
        return V, F


class UnetDecoder(nn.Module):
    def __init__(self):
        super(UnetDecoder, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.LeakyReLU(0.2, inplace=False)]

            cbr = nn.Sequential(*layers)

            return cbr

        self.dec_5_2 = CBR2d(in_channels=992+992, out_channels=256)        # 992+992，  256
        self.dec_5_1 = CBR2d(in_channels=256+256, out_channels=256)        # 256+256，  256
        self.uppool_4 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.dec_4_2 = CBR2d(in_channels=256+128, out_channels=128)        # 256+128，  128
        self.dec_4_1 = CBR2d(in_channels=128+128, out_channels=128)        # 128+128，  128
        self.uppool_3 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.dec_3_2 = CBR2d(in_channels=128+64, out_channels=64)          # 128+64，   64
        self.dec_3_1 = CBR2d(in_channels=64+64, out_channels=64)           # 64+64，    64
        self.uppool_2 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.dec_2_2 = CBR2d(in_channels=64+32, out_channels=32)           # 64+32     32
        self.dec_2_1 = CBR2d(in_channels=32+32, out_channels=32)           # 32+32     32
        self.uppool_1 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.dec_1_2 = CBR2d(in_channels=32+16, out_channels=16)           # 32+16     16
        self.dec_1_1 = CBR2d(in_channels=16+16, out_channels=16)           # 16+16     16

        self.fc = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, F):
        # x: [b, 992+992, 16, 16]
        dec_5_2 = self.dec_5_2(x)                                    # [b, 256, 16, 16]
        dec_5_1 = self.dec_5_1(torch.cat([dec_5_2, F[0]], dim=1))    # [b, 256, 16, 16]
        uppool_4 = self.uppool_4(dec_5_1)                            # [b, 256, 32, 32]

        dec_4_2 = self.dec_4_2(torch.cat([uppool_4, F[1]], dim=1))   # [b, 128, 32, 32]
        dec_4_1 = self.dec_4_1(torch.cat([dec_4_2, F[2]], dim=1))    # [b, 128, 32, 32]
        uppool_3 = self.uppool_3(dec_4_1)                            # [b, 128, 64, 64]

        dec_3_2 = self.dec_3_2(torch.cat([uppool_3, F[3]], dim=1))   # [b, 64, 64, 64]
        dec_3_1 = self.dec_3_1(torch.cat([dec_3_2, F[4]], dim=1))    # [b ,64, 64, 64]
        uppool_2 = self.uppool_2(dec_3_1)                            # [b ,64, 128, 128]

        dec_2_2 = self.dec_2_2(torch.cat([uppool_2, F[5]], dim=1))   # [b ,32, 128, 128]
        dec_2_1 = self.dec_2_1(torch.cat([dec_2_2, F[6]], dim=1))    # [b ,32, 128, 128]
        uppool_1 = self.uppool_1(dec_2_1)                            # [b ,32, 256, 256]

        dec_1_2 = self.dec_1_2(torch.cat([uppool_1, F[7]], dim=1))   # [b ,16, 256, 256]
        dec_1_1 = self.dec_1_1(torch.cat([dec_1_2, F[8]], dim=1))    # [b ,16, 256, 256]

        x = self.fc(dec_1_1)
        x = nn.Tanh()(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        def block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU(True)]
            layers += [nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            cbr = nn.Sequential(*layers)

            return cbr

        self.block_1 = block(in_channels, out_channels)
        self.block_2 = block(out_channels, out_channels)
        self.block_3 = block(out_channels, out_channels)
        self.block_4 = block(out_channels, out_channels)

        self.relu = nn.ReLU(True)

    def forward(self, x):

        residual = x
        out = self.block_1(x)
        out += residual
        out = self.relu(out)

        residual = out
        out = self.block_2(out)
        out += residual
        out = self.relu(out)

        residual = out
        out = self.block_3(out)
        out += residual
        out = self.relu(out)

        residual = out
        out = self.block_4(out)
        out += residual
        out = self.relu(out)

        return out


class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()

        def CL2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            layers += [nn.LeakyReLU(0.2)]
            cbr = nn.Sequential(*layers)

            return cbr

        # conv_layer
        self.conv_1 = CL2d(in_channels, 16)           # 256
        self.conv_2 = CL2d(16, 16)                    # 256
        self.conv_3 = CL2d(16, 32, stride=2)          # 128
        self.conv_4 = CL2d(32, 32)                    # 128
        self.conv_5 = CL2d(32, 64, stride=2)          # 64
        self.conv_6 = CL2d(64, 64)                    # 64
        self.conv_7 = CL2d(64, 128, stride=2)         # 32
        self.conv_8 = CL2d(128, 128)                  # 32
        self.conv_9 = CL2d(128, 256, stride=2)        # 16
        self.conv_10 = CL2d(256, 256)                 # 16

        # down_sample_layer
        self.down_sampling = nn.AdaptiveAvgPool2d((16, 16))


    def forward(self, x):
        f1 = self.conv_1(x)
        f2 = self.conv_2(f1)
        f3 = self.conv_3(f2)
        f4 = self.conv_4(f3)
        f5 = self.conv_5(f4)
        f6 = self.conv_6(f5)
        f7 = self.conv_7(f6)
        f8 = self.conv_8(f7)
        f9 = self.conv_9(f8)
        f10 = self.conv_10(f9)

        F = [f9, f8, f7, f6, f5, f4, f3, f2, f1]

        v1 = self.down_sampling(f1)
        v2 = self.down_sampling(f2)
        v3 = self.down_sampling(f3)
        v4 = self.down_sampling(f4)
        v5 = self.down_sampling(f5)
        v6 = self.down_sampling(f6)
        v7 = self.down_sampling(f7)
        v8 = self.down_sampling(f8)

        V = torch.cat([v1, v2, v3, v4, v5, v6, v7, v8, f9, f10], dim=1)
        h, w = V.size(2), V.size(3)
        V = V.view(V.size(0), V.size(1), h*w)
        V = torch.permute(V, [0, 2, 1])

        return V, F, (h, w)


class UnetDecoder(nn.Module):
    def __init__(self):
        super(UnetDecoder, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.LeakyReLU(0.2)]

            cbr = nn.Sequential(*layers)

            return cbr

        self.dec_5_2 = CBR2d(in_channels=992+992, out_channels=256)
        self.dec_5_1 = CBR2d(in_channels=256+256, out_channels=256)
        self.uppool_4 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.dec_4_2 = CBR2d(in_channels=256+128, out_channels=128)
        self.dec_4_1 = CBR2d(in_channels=128+128, out_channels=128)
        self.uppool_3 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.dec_3_2 = CBR2d(in_channels=128+64, out_channels=64)
        self.dec_3_1 = CBR2d(in_channels=64+64, out_channels=64)
        self.uppool_2 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.dec_2_2 = CBR2d(in_channels=64+32, out_channels=32)
        self.dec_2_1 = CBR2d(in_channels=32+32, out_channels=32)
        # self.uppool_1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0, bias=True)
        self.uppool_1 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.dec_1_2 = CBR2d(in_channels=32+16, out_channels=16)
        self.dec_1_1 = CBR2d(in_channels=16+16, out_channels=16)

        self.fc = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, F):
        # x: [b, 992+992, 16, 16]
        dec_5_2 = self.dec_5_2(x)                                    # [b, 256, 16, 16]
        dec_5_1 = self.dec_5_1(torch.cat([dec_5_2, F[0]], dim=1))    # [b, 256, 16, 16]
        uppool_4 = self.uppool_4(dec_5_1)                            # [b, 256, 32, 32]

        dec_4_2 = self.dec_4_2(torch.cat([uppool_4, F[1]], dim=1))   # [b, 128, 32, 32]
        dec_4_1 = self.dec_4_1(torch.cat([dec_4_2, F[2]], dim=1))    # [b, 128, 32, 32]
        uppool_3 = self.uppool_3(dec_4_1)                            # [b, 128, 64, 64]

        dec_3_2 = self.dec_3_2(torch.cat([uppool_3, F[3]], dim=1))   # [b, 64, 64, 64]
        dec_3_1 = self.dec_3_1(torch.cat([dec_3_2, F[4]], dim=1))    # [b ,64, 64, 64]
        uppool_2 = self.uppool_2(dec_3_1)                            # [b ,64, 128, 128]

        dec_2_2 = self.dec_2_2(torch.cat([uppool_2, F[5]], dim=1))   # [b ,32, 128, 128]
        dec_2_1 = self.dec_2_1(torch.cat([dec_2_2, F[6]], dim=1))    # [b ,32, 128, 128]
        uppool_1 = self.uppool_1(dec_2_1)                            # [b ,32, 256, 256]

        dec_1_2 = self.dec_1_2(torch.cat([uppool_1, F[7]], dim=1))   # [b ,16, 256, 256]
        dec_1_1 = self.dec_1_1(torch.cat([dec_1_2, F[8]], dim=1))    # [b ,16, 256, 256]

        x = self.fc(dec_1_1)
        x = nn.Tanh()(x)

        return x


class SCFT(nn.Module):
    def __init__(self, dv=992):
        super(SCFT, self).__init__()
        self.dv = torch.tensor(dv, dtype=torch.float32)

        self.w_q = nn.Linear(dv, dv)
        self.w_k = nn.Linear(dv, dv)
        self.w_v = nn.Linear(dv, dv)

    def forward(self, Vs, Vr, shape):
        # Vs Vr [b, 256, 992]
        h, w = shape
        query = self.w_q(Vs)
        key = self.w_k(Vr)
        value = self.w_v(Vr)

        c = torch.add(self.scaled_dot_product(query, key, value), Vs)
        c = torch.permute(c, [0, 2, 1])
        c = c.view(c.size(0), c.size(1), h, w)

        return c, query, key, value

    def scaled_dot_product(self, query, key, value):
        """
            Compute 'Scaled Dot Product Attention'
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.permute(0, 2, 1)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim=-1)

        return torch.matmul(p_attn, value)


class Generator_SCFT_cbam(nn.Module):
    def __init__(self, sketch_channels=1, reference_channels=3):
        super(Generator_SCFT_cbam, self).__init__()
        self.encoder_sketch = Encoder(sketch_channels)
        self.encoder_reference = Encoder(reference_channels)
        self.scft = SCFT()
        self.res_block = ResBlock(992, 992)
        self.unet_decoder = UnetDecoder()


    def forward(self, sketch_img, reference_img):
        Vs, F, shape = self.encoder_sketch(sketch_img)
        Vr, _, _ = self.encoder_reference(reference_img)

        c, query, key, value = self.scft(Vs, Vr, shape)
        c_out = self.res_block(c)
        img_gen = self.unet_decoder(torch.cat([c, c_out], dim=1), F)

        return img_gen, query, key, value


class Discriminator(nn.Module):
    # LSGAN + SN
    def __init__(self, ndf, in_channels):
        super(Discriminator, self).__init__()
        self.layer_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=False)
        )

        self.layer_2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=ndf, out_channels=ndf*2, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=False)
        )

        self.layer_3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=ndf*2, out_channels=ndf*4, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=False)
        )

        self.layer_4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=ndf*4, out_channels=ndf*8, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=False)
        )

        self.layer_5 = nn.Sequential(
            nn.Conv2d(in_channels=ndf*8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False)
        )


    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        return out
