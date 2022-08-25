from models.baseD import CCP_Generator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from functools import partial
from .drop import DropPath
from .weight_init import trunc_normal_  
import math
import copy


class SegNetEnc(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor,num_layers):
        super().__init__()

        layers = [
            nn.Upsample(scale_factor=scale_factor, mode='bilinear'),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            #nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            #nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ] * num_layers
        layers += [
            nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # self.dwconv = DWConv(hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

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
            # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out = 3 * 3 * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        # x = self.dwconv(x, H, W)

        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
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

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp3 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.norm21 = norm_layer(dim)
        self.attn2 = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm22 = norm_layer(dim)
        mlp_hidden_dim2 = int(dim * mlp_ratio)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim2, act_layer=act_layer, drop=drop)
        self.deconv = nn.Sequential(
            nn.Conv2d(dim, dim,3,1,1),
            #nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
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

    def forward(self, x, y, xy, H, W):
        # print("------------------------------------")
        # print("x.shape: {}".format(x.shape))
        # print("H: {}".format(H))
        # print("norm shape: {}".format(self.norm1(x).shape))
        # print("drop shape: {}".format(self.drop_path(self.attn(self.norm1(x), H, W)).shape))
        # print("------------------------------------")
        
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        y = y + self.drop_path(self.attn(self.norm1(y), H, W))
        y = y + self.drop_path(self.mlp(self.norm2(y), H, W))
        B,_,_ = x.shape
        # print("--------------------------------------")
        # print(x.transpose(2,1).reshape(B,-1,H,W).shape)
        # print(y.transpose(2,1).reshape(B,-1,H,W).shape)
        # print("--------------------------------------")
        # cp = self.mlp3(self.deconv(abs(x.transpose(2,1).reshape(B,-1,H,W)-y.transpose(2,1).reshape(B,-1,H,W))).flatten(2).transpose(2,1),H, W)
        ad = self.mlp3(self.norm3(abs(x-y)), H, W) 
        xy = xy + self.drop_path2(self.attn2(self.norm21(ad), H, W))
        xy = xy + self.drop_path2(self.mlp2(self.norm22(ad), H, W))

        return x, y, xy



class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x



class feature_extractor(nn.Module):
    def __init__(self, img_size=320, patch_size=4, in_chans=3, num_classes=2, embed_dims=[8,128, 256, 512, 512],
                 num_heads=[1, 1, 2, 4, 8], mlp_ratios=[2,2,2,2,2], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[2,2,2,2,2], sr_ratios=[16, 8, 4, 2, 1],feature_strides=[4, 8, 16, 32], embedding_dim =256, 
                 in_channels=[64, 128, 256, 512, 512]):

        super(feature_extractor, self).__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.in_channels = in_channels

        # Encoding Network
        vgg = models.vgg16_bn(pretrained=True)
        features = list(vgg.features.children())
        self.dec1 = nn.Sequential(*features[:7])  # 160
        self.dec2 = nn.Sequential(*features[7:14])  # 80
        self.dec3 = nn.Sequential(*features[14:24])  # 40
        self.dec4 = nn.Sequential(*features[24:34])  # 20
        self.dec5 = nn.Sequential(*features[34:44])  # 10

        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3,3), padding=1),
            #nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(512, 512, kernel_size=(3,3), padding=1),
            #nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=(3,3), padding=1),
            #nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(512, 512, kernel_size=(3,3), padding=1),
            #nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=(3,3), padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=(3,3), padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(128, 128, kernel_size=(3,3), padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=(3,3), padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(64, 64, kernel_size=(3,3), padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block0 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm0 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm1 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm2 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm3 = norm_layer(embed_dims[3])

        cur += depths[3]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[4], num_heads=num_heads[4], mlp_ratio=mlp_ratios[4], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[4])
            for i in range(depths[4])])
        self.norm4 = norm_layer(embed_dims[4])

        # self.feature_strides = feature_strides

        # c0_in_channels, c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # decoder_params = kwargs['decoder_params']
        # embedding_dim = decoder_params['embed_dim']

        # self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        # self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        # self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        # self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.conv_c4 = SegNetEnc(512,512,2,1)
        self.conv_c3 = SegNetEnc(1024,256,2,1)
        self.conv_c2 = SegNetEnc(512,128,2,1)
        self.conv_c1 = SegNetEnc(256,64,2,1)
        self.conv_c0 = SegNetEnc(72,8,1,1)


        # self.linear_fuse = ConvModule(
        #     in_channels=embedding_dim*4,
        #     out_channels=embedding_dim,
        #     kernel_size=1,
        #     norm_cfg=dict(type='BN', requires_grad=True)
        # )
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim*4, embedding_dim,3,1,1),
            #nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
    
        self.ccp_G = CCP_Generator(64,512)
    

        self.cd1 = nn.Sequential(
            nn.Conv2d(256,128,3,1,1),
            #nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.cd2 = nn.Sequential(
            nn.Conv2d(512,256,3,1,1),
            #nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.cd3 = nn.Sequential(
            nn.Conv2d(1024,512,3,1,1),
            #nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.cd4 = nn.Sequential(
            nn.Conv2d(1024,512,3,1,1),
            #nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.mp = nn.Conv2d(512,2,3,1,1)
        self.pf = nn.Conv2d(12,2,3,1,1)
   
        self.alpha0 = nn.Conv2d(512,8,1)
        self.alpha1 = nn.Conv2d(512,128,1)
        self.alpha2 = nn.Conv2d(512,256,1)
        self.alpha3 = nn.Conv2d(512,512,1)
        self.alpha4 = nn.Conv2d(512,512,1)
        self.f1_deconv = nn.Conv2d(64,8,1)
        self._c4 = nn.Conv2d(512,2,3,1,1)
        self._c3 = nn.Conv2d(256,2,3,1,1)
        self._c2 = nn.Conv2d(128,2,3,1,1)
        self._c1 = nn.Conv2d(64,2,3,1,1)
        self._c0 = nn.Conv2d(8,2,3,1,1)
        self.ad1 = nn.Sequential(
            nn.Conv2d(64,64,3,1,1),
         
            nn.ReLU(inplace=True),
            nn.Conv2d(64,2,1)
        )
        self.ad2 = nn.Sequential(
            nn.Conv2d(128,128,3,1,1),
           
            nn.ReLU(inplace=True),
            nn.Conv2d(128,2,1)
        )
        self.ad3 = nn.Sequential(
            nn.Conv2d(256,256,3,1,1),
         
            nn.ReLU(inplace=True),
            nn.Conv2d(256,2,1)
        )
        self.ad4 = nn.Sequential(
            nn.Conv2d(512,512,3,1,1),
          
            nn.ReLU(inplace=True),
            nn.Conv2d(512,2,1)
        )
        self.ad5 = nn.Sequential(
            nn.Conv2d(512,512,3,1,1),
     
            nn.ReLU(inplace=True),
            nn.Conv2d(512,2,1)
        )
     
        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

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

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            print("MMCV Error!!!")
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block0[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]
        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[3]
        for i in range(self.depths[4]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x,y):
        B = x.shape[0]
        outs = []
        x_f1 = self.dec1(x)
        x_f2 = self.dec2(x_f1)
        x_f3 = self.dec3(x_f2)
        x_f4 = self.dec4(x_f3)
        x_f5 = self.dec5(x_f4)
        y_f1 = self.dec1(y)
        y_f2 = self.dec2(y_f1)
        y_f3 = self.dec3(y_f2)
        y_f4 = self.dec4(y_f3)
        y_f5 = self.dec5(y_f4)

        ccp = self.ccp_G(x_f1, y_f1)


        x_f5 = self.enc5(x_f5)
        x_f4 = self.enc4(torch.cat([F.upsample_bilinear(x_f5, scale_factor=2), x_f4],1))
        x_f3 = self.enc3(torch.cat([F.upsample_bilinear(x_f4, scale_factor=2), x_f3],1))
        x_f2 = self.enc2(torch.cat([F.upsample_bilinear(x_f3, scale_factor=2), x_f2],1))
        x_f1 = self.enc1(torch.cat([F.upsample_bilinear(x_f2, scale_factor=2), x_f1],1))
        
        
        y_f5 = self.enc5(y_f5)
        y_f4 = self.enc4(torch.cat([F.upsample_bilinear(y_f5, scale_factor=2), y_f4],1))
        y_f3 = self.enc3(torch.cat([F.upsample_bilinear(y_f4, scale_factor=2), y_f3],1))
        y_f2 = self.enc2(torch.cat([F.upsample_bilinear(y_f3, scale_factor=2), y_f2],1))
        y_f1 = self.enc1(torch.cat([F.upsample_bilinear(y_f2, scale_factor=2), y_f1],1))


        ad1 = self.ad1(abs(y_f1-x_f1))
        ad2 = self.ad2(abs(y_f2-x_f2))
        ad3 = self.ad3(abs(y_f3-x_f3))
        ad4 = self.ad4(abs(y_f4-x_f4))
        ad5 = self.ad5(abs(y_f5-x_f5))
        ad = [ad1,ad2,ad3,ad4,ad5]



        x_f1 = self.f1_deconv(x_f1)
        y_f1 = self.f1_deconv(y_f1)
        ep1 = self.alpha0(F.upsample_bilinear(ccp, x_f1.size()[2:]))
        _, _, H,W = x_f1.shape
        ex1 = x_f1.flatten(2).transpose(1, 2)
        ey1 = y_f1.flatten(2).transpose(1, 2)
        ep1 = ep1.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block0):
            ex1,ey1,ep1 = blk(ex1,ey1,ep1, H, W)
         
          
        enc0 = self.norm0(ep1)
        enc0 = enc0.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        outs.append(enc0)

        ep2 = self.alpha1(F.upsample_bilinear(ccp, x_f2.size()[2:]))
        _, _, H,W = x_f2.shape
        ex2 = x_f2.flatten(2).transpose(1, 2)
        ey2 = y_f2.flatten(2).transpose(1, 2)
        ep2 = ep2.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block1):
            ex2,ey2,ep2 = blk(ex2,ey2,ep2, H, W)
          
        enc1 = self.norm1(ep2)
        enc1 = enc1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(enc1)

        ep3 = self.alpha2(F.upsample_bilinear(ccp, x_f3.size()[2:]))
        _, _, H,W = x_f3.shape
        ex3 = x_f3.flatten(2).transpose(1, 2)
        ey3 = y_f3.flatten(2).transpose(1, 2)
        ep3 = ep3.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block2):
            ex3,ey3,ep3 = blk(ex3,ey3,ep3, H, W)
            
        enc2 = self.norm2(ep3)
        enc2 = enc2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(enc2)
        ep4 = self.alpha3(F.upsample_bilinear(ccp, x_f4.size()[2:]))

        _, _, H,W = x_f4.shape
        ex4 = x_f4.flatten(2).transpose(1, 2)
        ey4 = y_f4.flatten(2).transpose(1, 2)
        ep4 = ep4.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block3):
            ex4,ey4,ep4 = blk(ex4,ey4,ep4, H, W)
         
        enc3 = self.norm3(ep4)
        enc3 = enc3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(enc3)

        ep5 = self.alpha4(F.upsample_bilinear(ccp, x_f5.size()[2:]))
        _, _, H,W = x_f5.shape
        ex5 = x_f5.flatten(2).transpose(1, 2)
        ey5 = y_f5.flatten(2).transpose(1, 2)
        ep5 = ep5.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block4):
            ex5,ey5,ep5 = blk(ex5,ey5,ep5, H, W)
       
        enc4 = self.norm4(ep5)
        enc4 = enc4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(enc4)

        return outs, ccp, ad

    def forward(self, x, y):

        out, ccp, ad = self.forward_features(x, y)

        c0, c1, c2, c3, c4 = out
        ad1, ad2, ad3, ad4, ad5 = ad



        _c4 = self.conv_c4(c4)
        _c3 = self.conv_c3(torch.cat([c3,_c4],1))
        _c2 = self.conv_c2(torch.cat([c2,_c3],1))
        _c1 = self.conv_c1(torch.cat([c1,_c2],1))
        _c0 = self.conv_c0(torch.cat([c0,_c1],1))

        _c4 = self._c4(_c4)
        _c3 = self._c3(_c3)
        _c2 = self._c2(_c2)
        _c1 = self._c1(_c1)
        _c0 = self._c0(_c0)
        mp = self.mp(ccp)
        mp = F.upsample_bilinear(mp, x.size()[2:])
        _c4 = F.upsample_bilinear(_c4, x.size()[2:])
        _c3 = F.upsample_bilinear(_c3, x.size()[2:])
        _c2 = F.upsample_bilinear(_c2, x.size()[2:])
        _c1 = F.upsample_bilinear(_c1, x.size()[2:])
        _c0 = F.upsample_bilinear(_c0, x.size()[2:])

        ad1 = F.upsample_bilinear(ad1, x.size()[2:])
        ad2 = F.upsample_bilinear(ad2, x.size()[2:])
        ad3 = F.upsample_bilinear(ad3, x.size()[2:])
        ad4 = F.upsample_bilinear(ad4, x.size()[2:])
        ad5 = F.upsample_bilinear(ad5, x.size()[2:])
        pf = self.pf(torch.cat([_c0, _c1, _c2, _c3, _c4, mp], 1))
        return mp, _c0, _c1, _c2, _c3, _c4, pf, ad1, ad2, ad3, ad4, ad5

        # return pf

