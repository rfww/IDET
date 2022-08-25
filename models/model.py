import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import math
from .weight_init import trunc_normal_


class Attention(nn.Module):
    # def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.3, proj_drop=0.3, sr_ratio=1):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y):
        # B, N, C = x.shape
        # q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        #
        # if self.sr_ratio > 1:
        #     x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        #     x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
        #     x_ = self.norm(x_)
        #     kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # else:
        #     kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # k, v = kv[0], kv[1]

        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        y = y.flatten(2).permute(0, 2, 1)

        _, N, _ = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(abs(x-y)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Attention2(nn.Module):
    # def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.3, proj_drop=0.3, sr_ratio=1):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super(Attention2, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y):
        # B, N, C = x.shape
        # q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        #
        # if self.sr_ratio > 1:
        #     x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        #     x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
        #     x_ = self.norm(x_)
        #     kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # else:
        #     kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # k, v = kv[0], kv[1]

        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        y = y.flatten(2).permute(0, 2, 1)

        _, N, _ = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # v = self.v(abs(x-y)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x+y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        # x = self.proj_drop(x)

        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        extractor = list(models.vgg16(pretrained=True).features.children())
        self.Enc1 = nn.Sequential(*extractor[:5])
        self.Enc2 = nn.Sequential(*extractor[5:10])
        self.Enc3 = nn.Sequential(*extractor[10:17])
        self.Enc4 = nn.Sequential(*extractor[17:24])
        self.Enc5 = nn.Sequential(*extractor[24:34])
        self.attention1 = Attention(512)
        self.attention2 = Attention(512)

        self.Dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.Dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.Dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.Dec4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.Dec5 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.diff = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )




    def forward(self, x, y):
        x_f1 = self.Enc1(x)
        x_f2 = self.Enc2(x_f1)
        x_f3 = self.Enc3(x_f2)
        x_f4 = self.Enc4(x_f3)
        x_f5 = self.Enc5(x_f4)

        y_f1 = self.Enc1(y)
        y_f2 = self.Enc2(y_f1)
        y_f3 = self.Enc3(y_f2)
        y_f4 = self.Enc4(y_f3)
        y_f5 = self.Enc5(y_f4)
        # print(y_f5.shape)
        # print(y_f4.shape)
        # print(y_f3.shape)
        # print(y_f2.shape)
        # print(y_f1.shape)
        B, C, H, W = y_f5.shape
        map1 = self.attention1(x_f5, 1-y_f5).permute(0, 2, 1).reshape(B, C, H, W)
        map2 = self.attention2(1-x_f5, y_f5).permute(0, 2, 1).reshape(B, C, H, W)

        x_d5 = self.Dec5(torch.cat([x_f5, map1], 1))
        # print(x_d5.shape)
        x_d4 = self.Dec4(torch.cat([x_f4, x_d5], 1))
        x_d3 = self.Dec3(torch.cat([x_f3, x_d4], 1))
        x_d2 = self.Dec2(torch.cat([x_f2, x_d3], 1))
        x_d1 = self.Dec1(torch.cat([x_f1, x_d2], 1))

        y_d5 = self.Dec5(torch.cat([y_f5, map2], 1))
        y_d4 = self.Dec4(torch.cat([y_f4, y_d5], 1))
        y_d3 = self.Dec3(torch.cat([y_f3, y_d4], 1))
        y_d2 = self.Dec2(torch.cat([y_f2, y_d3], 1))
        y_d1 = self.Dec1(torch.cat([y_f1, y_d2], 1))

        map = self.diff(abs(x_d1 - y_d1))

        # map = F.upsample_bilinear(map, x.size()[2:])
        map = F.interpolate(map, x.size()[2:], mode='bilinear')

        return map

    def init_weigth(self, Module):
        print()

class SegNetEnc(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor, num_layers):
        super().__init__()

        layers = [
            nn.Upsample(scale_factor=scale_factor, mode='bilinear'),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ] * num_layers
        layers += [
            nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class Generator2(nn.Module):
    def __init__(self):
        super(Generator2, self).__init__()
        extractor = list(models.vgg16(pretrained=True).features.children())
        self.Enc1 = nn.Sequential(*extractor[:5])
        self.Enc2 = nn.Sequential(*extractor[5:10])
        self.Enc3 = nn.Sequential(*extractor[10:17])
        self.Enc4 = nn.Sequential(*extractor[17:24])
        self.Enc5 = nn.Sequential(*extractor[24:34])
        self.attention1 = Attention2(512)
        self.attention2 = Attention2(512)

        # self.Dec1 = nn.Sequential(
        #     nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        # )
        # self.Dec2 = nn.Sequential(
        #     nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        # )
        # self.Dec3 = nn.Sequential(
        #     nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        # )
        # self.Dec4 = nn.Sequential(
        #     nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        # )
        # self.Dec5 = nn.Sequential(
        #     nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        # ) 

        self.Dec1 = SegNetEnc(128, 32, 2, 1)
        self.Dec2 = SegNetEnc(256, 64, 2, 1)
        self.Dec3 = SegNetEnc(512, 128, 2, 1)
        self.Dec4 = SegNetEnc(1024, 256, 2, 1)
        self.Dec5 = SegNetEnc(1024, 512, 2, 1)
        self.diff1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )
        self.diff2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )
        self.diff3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )
        self.diff4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )
        self.diff5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )
        self.diff6 = nn.Sequential(
            nn.Conv2d(10, 10, 3, 1, 1),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )




    def forward(self, x, y):
        x_f1 = self.Enc1(x)
        x_f2 = self.Enc2(x_f1)
        x_f3 = self.Enc3(x_f2)
        x_f4 = self.Enc4(x_f3)
        x_f5 = self.Enc5(x_f4)

        y_f1 = self.Enc1(y)
        y_f2 = self.Enc2(y_f1)
        y_f3 = self.Enc3(y_f2)
        y_f4 = self.Enc4(y_f3)
        y_f5 = self.Enc5(y_f4)
        # print(y_f5.shape)
        # print(y_f4.shape)
        # print(y_f3.shape)
        # print(y_f2.shape)
        # print(y_f1.shape)
        B, C, H, W = y_f5.shape
        map1 = self.attention1(x_f5, 1-y_f5).permute(0, 2, 1).reshape(B, C, H, W)
        map2 = self.attention2(1-x_f5, y_f5).permute(0, 2, 1).reshape(B, C, H, W)

        x_d5 = self.Dec5(torch.cat([x_f5, map1], 1))
        # print(x_d5.shape)
        x_d4 = self.Dec4(torch.cat([x_f4, x_d5], 1))
        x_d3 = self.Dec3(torch.cat([x_f3, x_d4], 1))
        x_d2 = self.Dec2(torch.cat([x_f2, x_d3], 1))
        x_d1 = self.Dec1(torch.cat([x_f1, x_d2], 1))

        y_d5 = self.Dec5(torch.cat([y_f5, map2], 1))
        y_d4 = self.Dec4(torch.cat([y_f4, y_d5], 1))
        y_d3 = self.Dec3(torch.cat([y_f3, y_d4], 1))
        y_d2 = self.Dec2(torch.cat([y_f2, y_d3], 1))
        y_d1 = self.Dec1(torch.cat([y_f1, y_d2], 1))

        map1 = self.diff1(abs(x_d1 - y_d1))
        map2 = self.diff2(abs(x_d2 - y_d2))
        map3 = self.diff3(abs(x_d3 - y_d3))
        map4 = self.diff4(abs(x_d4 - y_d4))
        map5 = self.diff5(abs(x_d5 - y_d5))


        

        # map = F.upsample_bilinear(map, x.size()[2:])
        map1 = F.interpolate(map1, x.size()[2:], mode='bilinear')
        map2 = F.interpolate(map2, x.size()[2:], mode='bilinear')
        map3 = F.interpolate(map3, x.size()[2:], mode='bilinear')
        map4 = F.interpolate(map4, x.size()[2:], mode='bilinear')
        map5 = F.interpolate(map5, x.size()[2:], mode='bilinear')
        map  = self.diff6(torch.cat([map1, map2, map3, map4, map5], 1))

        return map, map1, map2, map3, map4, map5
        # return map, map1

    def init_weigth(self, Module):
        print()



class Generator3(nn.Module):   # dissatisfy
    def __init__(self):
        super(Generator3, self).__init__()
        extractor = list(models.vgg16(pretrained=True).features.children())
        self.Enc1 = nn.Sequential(*extractor[:5])
        self.Enc2 = nn.Sequential(*extractor[5:10])
        self.Enc3 = nn.Sequential(*extractor[10:17])
        self.Enc4 = nn.Sequential(*extractor[17:24])
        self.Enc5 = nn.Sequential(*extractor[24:34])
        self.attention1 = Attention(512)
        self.attention2 = Attention(512)

        # self.Dec1 = nn.Sequential(
        #     nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        # )
        # self.Dec2 = nn.Sequential(
        #     nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        # )
        # self.Dec3 = nn.Sequential(
        #     nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        # )
        # self.Dec4 = nn.Sequential(
        #     nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        # )
        # self.Dec5 = nn.Sequential(
        #     nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        # ) 

        self.Dec1 = SegNetEnc(128, 32, 2, 1)
        self.Dec2 = SegNetEnc(256, 64, 2, 1)
        self.Dec3 = SegNetEnc(512, 128, 2, 1)
        self.Dec4 = SegNetEnc(1024, 256, 2, 1)
        self.Dec5 = SegNetEnc(1024, 512, 2, 1)
        self.diff1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32, 2, 1),
            # # nn.Sigmoid()
            # nn.ReLU(inplace=True)
        )
        self.diff2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(64, 2, 1),
            # # nn.Sigmoid()
            # nn.ReLU(inplace=True)
        )
        self.diff3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.Conv2d(128, 2, 1),
            # # nn.Sigmoid()
            # nn.ReLU(inplace=True)
        )
        self.diff4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 2, 1),
            # # nn.Sigmoid()
            # nn.ReLU(inplace=True)
        )
        self.diff5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # nn.Conv2d(512, 2, 1),
            # # nn.Sigmoid()
            # nn.ReLU(inplace=True)
        )
        self.diff6 = nn.Sequential(
            nn.Conv2d(10, 10, 3, 1, 1),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )

        self.fuse5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.fuse4 = nn.Sequential(
            nn.Conv2d(768, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(384, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(192, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(96, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.p5 = nn.Conv2d(512, 2, 1)
        self.p4 = nn.Conv2d(256, 2, 1)
        self.p3 = nn.Conv2d(128, 2, 1)
        self.p2 = nn.Conv2d(64, 2, 1)
        self.p1 = nn.Conv2d(32, 2, 1)




    def forward(self, x, y):
        x_f1 = self.Enc1(x)
        x_f2 = self.Enc2(x_f1)
        x_f3 = self.Enc3(x_f2)
        x_f4 = self.Enc4(x_f3)
        x_f5 = self.Enc5(x_f4)

        y_f1 = self.Enc1(y)
        y_f2 = self.Enc2(y_f1)
        y_f3 = self.Enc3(y_f2)
        y_f4 = self.Enc4(y_f3)
        y_f5 = self.Enc5(y_f4)
        # print(y_f5.shape)
        # print(y_f4.shape)
        # print(y_f3.shape)
        # print(y_f2.shape)
        # print(y_f1.shape)
        B, C, H, W = y_f5.shape
        map1 = self.attention1(x_f5, 1-y_f5).permute(0, 2, 1).reshape(B, C, H, W)
        map2 = self.attention2(1-x_f5, y_f5).permute(0, 2, 1).reshape(B, C, H, W)

        x_d5 = self.Dec5(torch.cat([x_f5, map1], 1))
        # print(x_d5.shape)
        x_d4 = self.Dec4(torch.cat([x_f4, x_d5], 1))
        x_d3 = self.Dec3(torch.cat([x_f3, x_d4], 1))
        x_d2 = self.Dec2(torch.cat([x_f2, x_d3], 1))
        x_d1 = self.Dec1(torch.cat([x_f1, x_d2], 1))

        y_d5 = self.Dec5(torch.cat([y_f5, map2], 1))
        y_d4 = self.Dec4(torch.cat([y_f4, y_d5], 1))
        y_d3 = self.Dec3(torch.cat([y_f3, y_d4], 1))
        y_d2 = self.Dec2(torch.cat([y_f2, y_d3], 1))
        y_d1 = self.Dec1(torch.cat([y_f1, y_d2], 1))

        map1 = self.diff1(abs(x_d1 - y_d1))
        map2 = self.diff2(abs(x_d2 - y_d2))
        map3 = self.diff3(abs(x_d3 - y_d3))
        map4 = self.diff4(abs(x_d4 - y_d4))
        map5 = self.diff5(abs(x_d5 - y_d5))

        map5 = self.fuse5(map5)
        map4 = self.fuse4(torch.cat([map4, F.upsample_bilinear(map5, scale_factor=2)], 1))
        map3 = self.fuse3(torch.cat([map3, F.upsample_bilinear(map4, scale_factor=2)], 1))
        map2 = self.fuse2(torch.cat([map2, F.upsample_bilinear(map3, scale_factor=2)], 1))
        map1 = self.fuse1(torch.cat([map1, F.upsample_bilinear(map2, scale_factor=2)], 1))

        map5 = self.p5(map5)
        map4 = self.p4(map4)
        map3 = self.p3(map3)
        map2 = self.p2(map2)
        map1 = self.p1(map1)


        

        # map = F.upsample_bilinear(map, x.size()[2:])
        map1 = F.interpolate(map1, x.size()[2:], mode='bilinear')
        map2 = F.interpolate(map2, x.size()[2:], mode='bilinear')
        map3 = F.interpolate(map3, x.size()[2:], mode='bilinear')
        map4 = F.interpolate(map4, x.size()[2:], mode='bilinear')
        map5 = F.interpolate(map5, x.size()[2:], mode='bilinear')
        map  = self.diff6(torch.cat([map1, map2, map3, map4, map5], 1))

        # return map, map1, map2, map3, map4, map5
        return map, map1

    def init_weigth(self, Module):
        print()



class Attention4(nn.Module):
    # def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.3, proj_drop=0.3, sr_ratio=1):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super(Attention4, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y, z):
        # B, N, C = x.shape
        # q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        #
        # if self.sr_ratio > 1:
        #     x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        #     x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
        #     x_ = self.norm(x_)
        #     kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # else:
        #     kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # k, v = kv[0], kv[1]

        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        y = y.flatten(2).permute(0, 2, 1)
        z = z.flatten(2).permute(0, 2, 1)

        _, N, _ = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(z).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class depth_conv(nn.Module):
    def __init__(self, C_in, C_out):
        super().__init__()
        self.diff = nn.Sequential(
            nn.Conv2d(C_in, C_in // 2, 1),
            nn.BatchNorm2d(C_in // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_in // 2, C_in // 2, 3, 1, 1),
            nn.BatchNorm2d(C_in // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_in // 2, C_out, 1),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.diff(x)


class Generator4(nn.Module):
    def __init__(self):
        super(Generator4, self).__init__()
        extractor = list(models.vgg16(pretrained=True).features.children())
        self.Enc1 = nn.Sequential(*extractor[:5])
        self.Enc2 = nn.Sequential(*extractor[5:10])
        self.Enc3 = nn.Sequential(*extractor[10:17])
        self.Enc4 = nn.Sequential(*extractor[17:24])
        self.Enc5 = nn.Sequential(*extractor[24:34])
        self.attention1 = Attention4(512)
        self.attention2 = Attention4(512)

        # self.Dec1 = nn.Sequential(
        #     nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        # )
        # self.Dec2 = nn.Sequential(
        #     nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        # )
        # self.Dec3 = nn.Sequential(
        #     nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        # )
        # self.Dec4 = nn.Sequential(
        #     nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        # )
        # self.Dec5 = nn.Sequential(
        #     nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        # ) 

        self.Dec1 = SegNetEnc(128, 32, 2, 1)
        self.Dec2 = SegNetEnc(256, 64, 2, 1)
        self.Dec3 = SegNetEnc(512, 128, 2, 1)
        self.Dec4 = SegNetEnc(1024, 256, 2, 1)
        self.Dec5 = SegNetEnc(1024, 512, 2, 1)
        self.diff1 = depth_conv(64, 64)
        self.diff2 = depth_conv(128, 128)
        self.diff3 = depth_conv(256, 256)
        self.diff4 = depth_conv(512, 512)
        self.diff5 = depth_conv(512, 512)

        self.ediff1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )
        self.ediff2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )
        self.ediff3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )
        self.ediff4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )
        self.ediff5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )
        # self.diff6 = nn.Sequential(
        #     nn.Conv2d(10, 10, 3, 1, 1),
        #     nn.BatchNorm2d(10),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(10, 2, 1),
        #     # nn.Sigmoid()
        #     nn.ReLU(inplace=True)
        # )

        self.p1 = nn.Conv2d(64, 2, 1)
        self.p2 = nn.Conv2d(128, 2, 1)
        self.p3 = nn.Conv2d(256, 2, 1)
        self.p4 = nn.Conv2d(512, 2, 1)
        self.p5 = nn.Conv2d(512, 2, 1)
        self.p6 = nn.Conv2d(10, 2, 1)




    def forward(self, x, y):
        x_f1 = self.Enc1(x)
        x_f2 = self.Enc2(x_f1)
        x_f3 = self.Enc3(x_f2)
        x_f4 = self.Enc4(x_f3)
        x_f5 = self.Enc5(x_f4)

        y_f1 = self.Enc1(y)
        y_f2 = self.Enc2(y_f1)
        y_f3 = self.Enc3(y_f2)
        y_f4 = self.Enc4(y_f3)
        y_f5 = self.Enc5(y_f4)
        # print(y_f5.shape)
        # print(y_f4.shape)
        # print(y_f3.shape)
        # print(y_f2.shape)
        # print(y_f1.shape)
        map1 = self.diff1(abs(x_f1 - y_f1))
        map2 = self.diff2(abs(x_f2 - y_f2))
        map3 = self.diff3(abs(x_f3 - y_f3))
        map4 = self.diff4(abs(x_f4 - y_f4))
        map5 = self.diff5(abs(x_f5 - y_f5))

 
        B, C, H, W = y_f5.shape
        mapa1 = self.attention1(x_f5, 1-y_f5, map5).permute(0, 2, 1).reshape(B, C, H, W)
        mapa2 = self.attention2(1-x_f5, y_f5, map5).permute(0, 2, 1).reshape(B, C, H, W)

        x_d5 = self.Dec5(torch.cat([x_f5, mapa1], 1))
        # print(x_d5.shape)
        x_d4 = self.Dec4(torch.cat([x_f4, x_d5], 1))
        x_d3 = self.Dec3(torch.cat([x_f3, x_d4], 1))
        x_d2 = self.Dec2(torch.cat([x_f2, x_d3], 1))
        x_d1 = self.Dec1(torch.cat([x_f1, x_d2], 1))

        y_d5 = self.Dec5(torch.cat([y_f5, mapa2], 1))
        y_d4 = self.Dec4(torch.cat([y_f4, y_d5], 1))
        y_d3 = self.Dec3(torch.cat([y_f3, y_d4], 1))
        y_d2 = self.Dec2(torch.cat([y_f2, y_d3], 1))
        y_d1 = self.Dec1(torch.cat([y_f1, y_d2], 1))



        map1 = self.p1(map1)
        map2 = self.p2(map2)
        map3 = self.p3(map3)
        map4 = self.p4(map4)
        map5 = self.p5(map5)

        mape1 = self.ediff1(abs(x_d1 - y_d1))
        mape2 = self.ediff2(abs(x_d2 - y_d2))
        mape3 = self.ediff3(abs(x_d3 - y_d3))
        mape4 = self.ediff4(abs(x_d4 - y_d4))
        mape5 = self.ediff5(abs(x_d5 - y_d5))
     

        # map = F.upsample_bilinear(map, x.size()[2:])
        map1 = F.interpolate(map1, x.size()[2:], mode='bilinear')
        map2 = F.interpolate(map2, x.size()[2:], mode='bilinear')
        map3 = F.interpolate(map3, x.size()[2:], mode='bilinear')
        map4 = F.interpolate(map4, x.size()[2:], mode='bilinear')
        map5 = F.interpolate(map5, x.size()[2:], mode='bilinear')
        mape1 = F.interpolate(mape1, x.size()[2:], mode='bilinear')
        mape2 = F.interpolate(mape2, x.size()[2:], mode='bilinear')
        mape3 = F.interpolate(mape3, x.size()[2:], mode='bilinear')
        mape4 = F.interpolate(mape4, x.size()[2:], mode='bilinear')
        mape5 = F.interpolate(mape5, x.size()[2:], mode='bilinear')
        map  = self.p6(torch.cat([mape1, mape2, mape3, mape4, mape5], 1))

        return map, map1, map2, map3, map4, map5, mape1, mape2, mape3, mape4, mape5
        # return map, map1

    def init_weigth(self, Module):
        print()





class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AvgPool2d((2, 2)),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((2, 2)),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(512, 1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        out = self.net(torch.cat([x, y], 1))
        return out


    def weight_init(self):
        print()



class MaskAttention(nn.Module):
    # def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.3, proj_drop=0.3, sr_ratio=1):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., mask_ratio=0.75):
        super(MaskAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.mask_ratio = mask_ratio
 

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y, z):
        # B, N, C = x.shape
        # q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        #
        # if self.sr_ratio > 1:
        #     x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        #     x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
        #     x_ = self.norm(x_)
        #     kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # else:
        #     kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # k, v = kv[0], kv[1]

        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        y = y.flatten(2).permute(0, 2, 1)
        z = z.flatten(2).permute(0, 2, 1)

        _, N, _ = x.shape
        # print(x.shape)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        mask_k, masked_k, ids_k = self.random_masking(y, self.mask_ratio)
        # print("-------------------------")
        # print(y.shape)
        # print(mask_k.shape)
        # print(masked_k.shape)
        # print(ids_k.shape)
        # print("-------------------------")
        NN = int(N * (1 - self.mask_ratio))
        k = self.k(mask_k).reshape(B, NN, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        mask_v, masked_v, ids_v = self.random_masking(z, self.mask_ratio)
        v = self.v(mask_v).reshape(B, NN, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


    def random_masking(self, x, mask_ratio):
        # print("***************************************")
        # print("mask ratio: {}".format(mask_ratio))
        # print("***************************************")
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        if mask_ratio == 0.:
            return x, None, None 
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore



class Generator5(nn.Module):
    def __init__(self):
        super(Generator5, self).__init__()
        extractor = list(models.vgg16(pretrained=True).features.children())
        self.Enc1 = nn.Sequential(*extractor[:5])
        self.Enc2 = nn.Sequential(*extractor[5:10])
        self.Enc3 = nn.Sequential(*extractor[10:17])
        self.Enc4 = nn.Sequential(*extractor[17:24])
        self.Enc5 = nn.Sequential(*extractor[24:34])
        self.attention1 = MaskAttention(512)
        self.attention2 = MaskAttention(512)

        # self.Dec1 = nn.Sequential(
        #     nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        # )
        # self.Dec2 = nn.Sequential(
        #     nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        # )
        # self.Dec3 = nn.Sequential(
        #     nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        # )
        # self.Dec4 = nn.Sequential(
        #     nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        # )
        # self.Dec5 = nn.Sequential(
        #     nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        # ) 

        self.Dec1 = SegNetEnc(128, 32, 2, 1)
        self.Dec2 = SegNetEnc(256, 64, 2, 1)
        self.Dec3 = SegNetEnc(512, 128, 2, 1)
        self.Dec4 = SegNetEnc(1024, 256, 2, 1)
        self.Dec5 = SegNetEnc(1024, 512, 2, 1)
        self.diff1 = depth_conv(64, 64)
        self.diff2 = depth_conv(128, 128)
        self.diff3 = depth_conv(256, 256)
        self.diff4 = depth_conv(512, 512)
        self.diff5 = depth_conv(512, 512)

        self.ediff1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )
        self.ediff2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )
        self.ediff3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )
        self.ediff4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )
        self.ediff5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )
        # self.diff6 = nn.Sequential(
        #     nn.Conv2d(10, 10, 3, 1, 1),
        #     nn.BatchNorm2d(10),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(10, 2, 1),
        #     # nn.Sigmoid()
        #     nn.ReLU(inplace=True)
        # )

        self.p1 = nn.Conv2d(64, 2, 1)
        self.p2 = nn.Conv2d(128, 2, 1)
        self.p3 = nn.Conv2d(256, 2, 1)
        self.p4 = nn.Conv2d(512, 2, 1)
        self.p5 = nn.Conv2d(512, 2, 1)
        self.p6 = nn.Conv2d(10, 2, 1)




    def forward(self, x, y):
        x_f1 = self.Enc1(x)
        x_f2 = self.Enc2(x_f1)
        x_f3 = self.Enc3(x_f2)
        x_f4 = self.Enc4(x_f3)
        x_f5 = self.Enc5(x_f4)

        y_f1 = self.Enc1(y)
        y_f2 = self.Enc2(y_f1)
        y_f3 = self.Enc3(y_f2)
        y_f4 = self.Enc4(y_f3)
        y_f5 = self.Enc5(y_f4)
        # print(y_f5.shape)
        # print(y_f4.shape)
        # print(y_f3.shape)
        # print(y_f2.shape)
        # print(y_f1.shape)
        map1 = self.diff1(abs(x_f1 - y_f1))
        map2 = self.diff2(abs(x_f2 - y_f2))
        map3 = self.diff3(abs(x_f3 - y_f3))
        map4 = self.diff4(abs(x_f4 - y_f4))
        map5 = self.diff5(abs(x_f5 - y_f5))

 
        B, C, H, W = y_f5.shape
        mapa1 = self.attention1(x_f5, 1-y_f5, map5).permute(0, 2, 1).reshape(B, C, H, W)
        mapa2 = self.attention2(1-x_f5, y_f5, map5).permute(0, 2, 1).reshape(B, C, H, W)

        x_d5 = self.Dec5(torch.cat([x_f5, mapa1], 1))
        # print(x_d5.shape)
        x_d4 = self.Dec4(torch.cat([x_f4, x_d5], 1))
        x_d3 = self.Dec3(torch.cat([x_f3, x_d4], 1))
        x_d2 = self.Dec2(torch.cat([x_f2, x_d3], 1))
        x_d1 = self.Dec1(torch.cat([x_f1, x_d2], 1))

        y_d5 = self.Dec5(torch.cat([y_f5, mapa2], 1))
        y_d4 = self.Dec4(torch.cat([y_f4, y_d5], 1))
        y_d3 = self.Dec3(torch.cat([y_f3, y_d4], 1))
        y_d2 = self.Dec2(torch.cat([y_f2, y_d3], 1))
        y_d1 = self.Dec1(torch.cat([y_f1, y_d2], 1))



        map1 = self.p1(map1)
        map2 = self.p2(map2)
        map3 = self.p3(map3)
        map4 = self.p4(map4)
        map5 = self.p5(map5)

        mape1 = self.ediff1(abs(x_d1 - y_d1))
        mape2 = self.ediff2(abs(x_d2 - y_d2))
        mape3 = self.ediff3(abs(x_d3 - y_d3))
        mape4 = self.ediff4(abs(x_d4 - y_d4))
        mape5 = self.ediff5(abs(x_d5 - y_d5))
     

        # map = F.upsample_bilinear(map, x.size()[2:])
        map1 = F.interpolate(map1, x.size()[2:], mode='bilinear')
        map2 = F.interpolate(map2, x.size()[2:], mode='bilinear')
        map3 = F.interpolate(map3, x.size()[2:], mode='bilinear')
        map4 = F.interpolate(map4, x.size()[2:], mode='bilinear')
        map5 = F.interpolate(map5, x.size()[2:], mode='bilinear')
        mape1 = F.interpolate(mape1, x.size()[2:], mode='bilinear')
        mape2 = F.interpolate(mape2, x.size()[2:], mode='bilinear')
        mape3 = F.interpolate(mape3, x.size()[2:], mode='bilinear')
        mape4 = F.interpolate(mape4, x.size()[2:], mode='bilinear')
        mape5 = F.interpolate(mape5, x.size()[2:], mode='bilinear')
        map  = self.p6(torch.cat([mape1, mape2, mape3, mape4, mape5], 1))

        return map, map1, map2, map3, map4, map5, mape1, mape2, mape3, mape4, mape5
        # return map, map1

    def init_weigth(self, Module):
        print()



class Generator6(nn.Module):
    def __init__(self):
        super(Generator6, self).__init__()
        extractor = list(models.vgg16(pretrained=True).features.children())
        self.Enc1 = nn.Sequential(*extractor[:5])
        self.Enc2 = nn.Sequential(*extractor[5:10])
        self.Enc3 = nn.Sequential(*extractor[10:17])
        self.Enc4 = nn.Sequential(*extractor[17:24])
        self.Enc5 = nn.Sequential(*extractor[24:34])
        # self.attention11 = MaskAttention(dim=32, mask_ratio=0.875)
        # self.attention12 = MaskAttention(dim=32, mask_ratio=0.875)

        self.attention21 = MaskAttention(dim=64, mask_ratio=0.75)
        self.attention22 = MaskAttention(dim=64, mask_ratio=0.75)

        self.attention31 = MaskAttention(dim=128, mask_ratio=0.75)
        self.attention32 = MaskAttention(dim=128, mask_ratio=0.75)

        self.attention41 = MaskAttention(dim=256, mask_ratio=0.5)
        self.attention42 = MaskAttention(dim=256, mask_ratio=0.5)

        self.attention51 = MaskAttention(dim=512, mask_ratio=0.25)
        self.attention52 = MaskAttention(dim=512, mask_ratio=0.25)

        self.Dec1 = SegNetEnc(96, 32, 2, 1)
        self.Dec2 = SegNetEnc(256, 64, 2, 1)
        self.Dec3 = SegNetEnc(512, 128, 2, 1)
        self.Dec4 = SegNetEnc(1024, 256, 2, 1)
        self.Dec5 = SegNetEnc(1024, 512, 2, 1)
        self.diff1 = depth_conv(64, 32)
        self.diff2 = depth_conv(128, 64)
        self.diff3 = depth_conv(256, 128)
        self.diff4 = depth_conv(512, 256)
        self.diff5 = depth_conv(512, 512)

        self.Com1 = depth_conv(64, 32)
        self.Com2 = depth_conv(128, 64)
        self.Com3 = depth_conv(256, 128)
        self.Com4 = depth_conv(512, 256)
        self.Com5 = depth_conv(512, 512)

        self.ediff1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )
        self.ediff2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )
        self.ediff3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )
        self.ediff4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )
        self.ediff5 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )
        # self.diff6 = nn.Sequential(
        #     nn.Conv2d(10, 10, 3, 1, 1),
        #     nn.BatchNorm2d(10),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(10, 2, 1),
        #     # nn.Sigmoid()
        #     nn.ReLU(inplace=True)
        # )

        self.p1 = nn.Conv2d(32, 2, 1)
        self.p2 = nn.Conv2d(64, 2, 1)
        self.p3 = nn.Conv2d(128, 2, 1)
        self.p4 = nn.Conv2d(256, 2, 1)
        self.p5 = nn.Conv2d(512, 2, 1)
        self.p6 = nn.Conv2d(10, 2, 1)




    def forward(self, x, y):
        x_f1 = self.Enc1(x)
        x_f2 = self.Enc2(x_f1)
        x_f3 = self.Enc3(x_f2)
        x_f4 = self.Enc4(x_f3)
        x_f5 = self.Enc5(x_f4)

        y_f1 = self.Enc1(y)
        y_f2 = self.Enc2(y_f1)
        y_f3 = self.Enc3(y_f2)
        y_f4 = self.Enc4(y_f3)
        y_f5 = self.Enc5(y_f4)
        # print(y_f5.shape)
        # print(y_f4.shape)
        # print(y_f3.shape)
        # print(y_f2.shape)
        # print(y_f1.shape)
        map1 = self.diff1(abs(x_f1 - y_f1))
        map2 = self.diff2(abs(x_f2 - y_f2))
        map3 = self.diff3(abs(x_f3 - y_f3))
        map4 = self.diff4(abs(x_f4 - y_f4))
        map5 = self.diff5(abs(x_f5 - y_f5))
        
        x_f5 = self.Com5(x_f5)
        x_f4 = self.Com4(x_f4)
        x_f3 = self.Com3(x_f3)
        x_f2 = self.Com2(x_f2)
        x_f1 = self.Com1(x_f1)

        y_f5 = self.Com5(y_f5)
        y_f4 = self.Com4(y_f4)
        y_f3 = self.Com3(y_f3)
        y_f2 = self.Com2(y_f2)
        y_f1 = self.Com1(y_f1)



 
        # B, C, H, W = y_f1.shape
        # mapa11 = self.attention11(x_f1, 1-y_f1, map1).permute(0, 2, 1).reshape(B, C, H, W)
        # mapa12 = self.attention12(1-x_f1, y_f1, map1).permute(0, 2, 1).reshape(B, C, H, W)
 
        B, C, H, W = y_f2.shape
        mapa21 = self.attention21(x_f2, 1-y_f2, map2).permute(0, 2, 1).reshape(B, C, H, W)
        mapa22 = self.attention22(1-x_f2, y_f2, map2).permute(0, 2, 1).reshape(B, C, H, W)
 
        B, C, H, W = y_f3.shape
        mapa31 = self.attention31(x_f3, 1-y_f3, map3).permute(0, 2, 1).reshape(B, C, H, W)
        mapa32 = self.attention32(1-x_f3, y_f3, map3).permute(0, 2, 1).reshape(B, C, H, W)
 
        B, C, H, W = y_f4.shape
        mapa41 = self.attention41(x_f4, 1-y_f4, map4).permute(0, 2, 1).reshape(B, C, H, W)
        mapa42 = self.attention42(1-x_f4, y_f4, map4).permute(0, 2, 1).reshape(B, C, H, W)
 
        B, C, H, W = y_f5.shape
        mapa51 = self.attention51(x_f5, 1-y_f5, map5).permute(0, 2, 1).reshape(B, C, H, W)
        mapa52 = self.attention52(1-x_f5, y_f5, map5).permute(0, 2, 1).reshape(B, C, H, W)

        x_d5 = self.Dec5(torch.cat([x_f5, mapa51], 1))
        x_d4 = self.Dec4(torch.cat([x_f4, x_d5, mapa41], 1))
        x_d3 = self.Dec3(torch.cat([x_f3, x_d4, mapa31], 1))
        # x_d2 = self.Dec2(torch.cat([x_f2, x_d3], 1))
        # x_d1 = self.Dec1(torch.cat([x_f1, x_d2], 1))
        x_d2 = self.Dec2(torch.cat([x_f2, x_d3, mapa21], 1))
        x_d1 = self.Dec1(torch.cat([x_f1, x_d2], 1))
        # x_d1 = self.Dec1(torch.cat([x_f1, x_d2, mapa11], 1))

        y_d5 = self.Dec5(torch.cat([y_f5, mapa52], 1))
        y_d4 = self.Dec4(torch.cat([y_f4, y_d5, mapa42], 1))
        y_d3 = self.Dec3(torch.cat([y_f3, y_d4, mapa32], 1))
        # y_d2 = self.Dec2(torch.cat([y_f2, y_d3], 1))
        # y_d1 = self.Dec1(torch.cat([y_f1, y_d2], 1))
        y_d2 = self.Dec2(torch.cat([y_f2, y_d3, mapa22], 1))
        y_d1 = self.Dec1(torch.cat([y_f1, y_d2], 1))
        # y_d1 = self.Dec1(torch.cat([y_f1, y_d2, mapa12], 1))

        # y_d5 = self.Dec5(torch.cat([y_f5, mapa52], 1))
        # y_d4 = self.Dec4(torch.cat([y_f4, y_d5, mapa42], 1))
        # y_d3 = self.Dec3(torch.cat([y_f3, y_d4, mapa32], 1))
        # y_d2 = self.Dec2(torch.cat([y_f2, y_d3, mapa22], 1))
        # y_d1 = self.Dec1(torch.cat([y_f1, y_d2, mapa12], 1))



        map1 = self.p1(map1)
        map2 = self.p2(map2)
        map3 = self.p3(map3)
        map4 = self.p4(map4)
        map5 = self.p5(map5)

        mape1 = self.ediff1(torch.cat([x_d1, y_d1], 1))
        mape2 = self.ediff2(torch.cat([x_d2, y_d2], 1))
        mape3 = self.ediff3(torch.cat([x_d3, y_d3], 1))
        mape4 = self.ediff4(torch.cat([x_d4, y_d4], 1))
        mape5 = self.ediff5(torch.cat([x_d5, y_d5], 1))
     

        # map = F.upsample_bilinear(map, x.size()[2:])
        map1 = F.interpolate(map1, x.size()[2:], mode='bilinear')
        map2 = F.interpolate(map2, x.size()[2:], mode='bilinear')
        map3 = F.interpolate(map3, x.size()[2:], mode='bilinear')
        map4 = F.interpolate(map4, x.size()[2:], mode='bilinear')
        map5 = F.interpolate(map5, x.size()[2:], mode='bilinear')
        mape1 = F.interpolate(mape1, x.size()[2:], mode='bilinear')
        mape2 = F.interpolate(mape2, x.size()[2:], mode='bilinear')
        mape3 = F.interpolate(mape3, x.size()[2:], mode='bilinear')
        mape4 = F.interpolate(mape4, x.size()[2:], mode='bilinear')
        mape5 = F.interpolate(mape5, x.size()[2:], mode='bilinear')
        map  = self.p6(torch.cat([mape1, mape2, mape3, mape4, mape5], 1))

        # return map, map1, map2, map3, map4, map5, mape1, mape2, mape3, mape4, mape5
        return map

    def init_weigth(self, Module):
        print()


class Generator7(nn.Module):
    def __init__(self):
        super(Generator7, self).__init__()
        extractor = list(models.vgg16(pretrained=True).features.children())
        self.Enc1 = nn.Sequential(*extractor[:5])
        self.Enc2 = nn.Sequential(*extractor[5:10])
        self.Enc3 = nn.Sequential(*extractor[10:17])
        self.Enc4 = nn.Sequential(*extractor[17:24])
        self.Enc5 = nn.Sequential(*extractor[24:34])
        # self.attention11 = MaskAttention(dim=32, mask_ratio=0.875)
        # self.attention12 = MaskAttention(dim=32, mask_ratio=0.875)

        self.attention21 = MaskAttention(dim=64, mask_ratio=0.75)
        self.attention22 = MaskAttention(dim=64, mask_ratio=0.75)

        self.attention31 = MaskAttention(dim=128, mask_ratio=0.5)
        self.attention32 = MaskAttention(dim=128, mask_ratio=0.5)

        self.attention41 = MaskAttention(dim=256, mask_ratio=0.25)
        self.attention42 = MaskAttention(dim=256, mask_ratio=0.25)

        self.attention51 = MaskAttention(dim=512, mask_ratio=0.)
        self.attention52 = MaskAttention(dim=512, mask_ratio=0.)

        self.Dec1 = SegNetEnc(96, 32, 2, 1)
        self.Dec2 = SegNetEnc(256, 64, 2, 1)
        self.Dec3 = SegNetEnc(512, 128, 2, 1)
        self.Dec4 = SegNetEnc(1024, 256, 2, 1)
        self.Dec5 = SegNetEnc(1024, 512, 2, 1)
        self.diff1 = depth_conv(64, 32)
        self.diff2 = depth_conv(128, 64)
        self.diff3 = depth_conv(256, 128)
        self.diff4 = depth_conv(512, 256)
        self.diff5 = depth_conv(512, 512)

        self.Com1 = depth_conv(64, 32)
        self.Com2 = depth_conv(128, 64)
        self.Com3 = depth_conv(256, 128)
        self.Com4 = depth_conv(512, 256)
        self.Com5 = depth_conv(512, 512)

        self.ediff1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )
        self.ediff2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )
        self.ediff3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )
        self.ediff4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )
        self.ediff5 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2, 1),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )
        # self.diff6 = nn.Sequential(
        #     nn.Conv2d(10, 10, 3, 1, 1),
        #     nn.BatchNorm2d(10),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(10, 2, 1),
        #     # nn.Sigmoid()
        #     nn.ReLU(inplace=True)
        # )

        self.p1 = nn.Conv2d(32, 2, 1)
        self.p2 = nn.Conv2d(64, 2, 1)
        self.p3 = nn.Conv2d(128, 2, 1)
        self.p4 = nn.Conv2d(256, 2, 1)
        self.p5 = nn.Conv2d(512, 2, 1)
        self.p6 = nn.Conv2d(10, 2, 1)




    def forward(self, x, y):
        x_f1 = self.Enc1(x)
        x_f2 = self.Enc2(x_f1)
        x_f3 = self.Enc3(x_f2)
        x_f4 = self.Enc4(x_f3)
        x_f5 = self.Enc5(x_f4)

        y_f1 = self.Enc1(y)
        y_f2 = self.Enc2(y_f1)
        y_f3 = self.Enc3(y_f2)
        y_f4 = self.Enc4(y_f3)
        y_f5 = self.Enc5(y_f4)
        # print(y_f5.shape)
        # print(y_f4.shape)
        # print(y_f3.shape)
        # print(y_f2.shape)
        # print(y_f1.shape)
        map1 = self.diff1(abs(x_f1 - y_f1))
        map2 = self.diff2(abs(x_f2 - y_f2))
        map3 = self.diff3(abs(x_f3 - y_f3))
        map4 = self.diff4(abs(x_f4 - y_f4))
        map5 = self.diff5(abs(x_f5 - y_f5))
        
        x_f5 = self.Com5(x_f5)
        x_f4 = self.Com4(x_f4)
        x_f3 = self.Com3(x_f3)
        x_f2 = self.Com2(x_f2)
        x_f1 = self.Com1(x_f1)

        y_f5 = self.Com5(y_f5)
        y_f4 = self.Com4(y_f4)
        y_f3 = self.Com3(y_f3)
        y_f2 = self.Com2(y_f2)
        y_f1 = self.Com1(y_f1)



 
        # B, C, H, W = y_f1.shape
        # mapa11 = self.attention11(x_f1, 1-y_f1, map1).permute(0, 2, 1).reshape(B, C, H, W)
        # mapa12 = self.attention12(1-x_f1, y_f1, map1).permute(0, 2, 1).reshape(B, C, H, W)
 
        B, C, H, W = y_f2.shape
        mapa21 = self.attention21(x_f2, 1-y_f2, map2).permute(0, 2, 1).reshape(B, C, H, W)
        mapa22 = self.attention22(1-x_f2, y_f2, map2).permute(0, 2, 1).reshape(B, C, H, W)
 
        B, C, H, W = y_f3.shape
        mapa31 = self.attention31(x_f3, 1-y_f3, map3).permute(0, 2, 1).reshape(B, C, H, W)
        mapa32 = self.attention32(1-x_f3, y_f3, map3).permute(0, 2, 1).reshape(B, C, H, W)
 
        B, C, H, W = y_f4.shape
        mapa41 = self.attention41(x_f4, 1-y_f4, map4).permute(0, 2, 1).reshape(B, C, H, W)
        mapa42 = self.attention42(1-x_f4, y_f4, map4).permute(0, 2, 1).reshape(B, C, H, W)
 
        B, C, H, W = y_f5.shape
        mapa51 = self.attention51(x_f5, 1-y_f5, map5).permute(0, 2, 1).reshape(B, C, H, W)
        mapa52 = self.attention52(1-x_f5, y_f5, map5).permute(0, 2, 1).reshape(B, C, H, W)

        x_d5 = self.Dec5(torch.cat([x_f5, mapa51], 1))
        x_d4 = self.Dec4(torch.cat([x_f4, x_d5, mapa41], 1))
        x_d3 = self.Dec3(torch.cat([x_f3, x_d4, mapa31], 1))
        # x_d2 = self.Dec2(torch.cat([x_f2, x_d3], 1))
        # x_d1 = self.Dec1(torch.cat([x_f1, x_d2], 1))
        x_d2 = self.Dec2(torch.cat([x_f2, x_d3, mapa21], 1))
        x_d1 = self.Dec1(torch.cat([x_f1, x_d2], 1))
        # x_d1 = self.Dec1(torch.cat([x_f1, x_d2, mapa11], 1))

        y_d5 = self.Dec5(torch.cat([y_f5, mapa52], 1))
        y_d4 = self.Dec4(torch.cat([y_f4, y_d5, mapa42], 1))
        y_d3 = self.Dec3(torch.cat([y_f3, y_d4, mapa32], 1))
        # y_d2 = self.Dec2(torch.cat([y_f2, y_d3], 1))
        # y_d1 = self.Dec1(torch.cat([y_f1, y_d2], 1))
        y_d2 = self.Dec2(torch.cat([y_f2, y_d3, mapa22], 1))
        y_d1 = self.Dec1(torch.cat([y_f1, y_d2], 1))
        # y_d1 = self.Dec1(torch.cat([y_f1, y_d2, mapa12], 1))

        # y_d5 = self.Dec5(torch.cat([y_f5, mapa52], 1))
        # y_d4 = self.Dec4(torch.cat([y_f4, y_d5, mapa42], 1))
        # y_d3 = self.Dec3(torch.cat([y_f3, y_d4, mapa32], 1))
        # y_d2 = self.Dec2(torch.cat([y_f2, y_d3, mapa22], 1))
        # y_d1 = self.Dec1(torch.cat([y_f1, y_d2, mapa12], 1))



        map1 = self.p1(map1)
        map2 = self.p2(map2)
        map3 = self.p3(map3)
        map4 = self.p4(map4)
        map5 = self.p5(map5)

        mape1 = self.ediff1(torch.cat([x_d1, y_d1], 1))
        mape2 = self.ediff2(torch.cat([x_d2, y_d2], 1))
        mape3 = self.ediff3(torch.cat([x_d3, y_d3], 1))
        mape4 = self.ediff4(torch.cat([x_d4, y_d4], 1))
        mape5 = self.ediff5(torch.cat([x_d5, y_d5], 1))
     

        # map = F.upsample_bilinear(map, x.size()[2:])
        map1 = F.interpolate(map1, x.size()[2:], mode='bilinear')
        map2 = F.interpolate(map2, x.size()[2:], mode='bilinear')
        map3 = F.interpolate(map3, x.size()[2:], mode='bilinear')
        map4 = F.interpolate(map4, x.size()[2:], mode='bilinear')
        map5 = F.interpolate(map5, x.size()[2:], mode='bilinear')
        mape1 = F.interpolate(mape1, x.size()[2:], mode='bilinear')
        mape2 = F.interpolate(mape2, x.size()[2:], mode='bilinear')
        mape3 = F.interpolate(mape3, x.size()[2:], mode='bilinear')
        mape4 = F.interpolate(mape4, x.size()[2:], mode='bilinear')
        mape5 = F.interpolate(mape5, x.size()[2:], mode='bilinear')
        map  = self.p6(torch.cat([mape1, mape2, mape3, mape4, mape5], 1))

        # return map, map1, map2, map3, map4, map5, mape1, mape2, mape3, mape4, mape5
        return map, mape1, mape2, mape3, mape4, mape5

    def init_weigth(self, Module):
        print()






























