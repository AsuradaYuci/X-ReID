import torch
import torch.nn as nn
from .vit_pytorch import DropPath
from .vit_pytorch import Mlp
from .vit_pytorch import trunc_normal_
from .vit_pytorch import Attention


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.normy = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, C = y.shape
        q = self.q_(x).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_(self.normy(y)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # torch.Size([20, 12, 1, 162])
        x = (attn @ v).transpose(1, 2)  # torch.Size([20, 1, 12, 64])
        x = x.reshape(B, C)  # torch.Size([20, 768])
        x = self.proj(x)
        x = self.proj_drop(x)  # torch.Size([32, 768])
        return x


class RotationAttention(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y):
        x = x + self.drop_path(self.attn(self.norm1(x), y))
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # torch.Size([32, 768])
        return x


class BlockRotation(nn.Module):

    def __init__(self, dim, num_heads, mode=0):
        super().__init__()
        self.Rotation = RotationAttention(dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                                          attn_drop=0.,
                                          drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.mode = mode

    def forward(self, x, y):  # torch.Size([32, 129, 768])
        x_cls = self.Rotation(x, y)  # torch.Size([20, 768])
        return x_cls


class BlockRotation_self(nn.Module):

    def __init__(self, dim, num_heads, mode=0):
        super().__init__()
        self.Rotation = RotationAttention(dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                                          attn_drop=0.,
                                          drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.mode = mode

    def forward(self, x, y):  # torch.Size([32, 129, 768])
        x_cls = self.Rotation(x, y)
        return x_cls


class ReUnit(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ReBlock(nn.Module):

    def __init__(self, dim, num_heads, depth=1, mode=0):
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList()
        self.mode = mode
        for i in range(self.depth):
            self.blocks.append(
                ReUnit(dim, num_heads, qkv_bias=False, qk_scale=None, drop=0.,
                       attn_drop=0.,
                       drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class TPM_temp_shift(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        # self.Reconstuct_shift = ReBlock(dim, num_heads)
        self.Reconstuct_shift = BlockRotation(dim, num_heads)
        # self.Ro_end = BlockRotation(dim, num_heads, mode=1)
        # self.apply(self._init_weights)  change to random
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, B, T):  # x: torch.Size([40, 163, 768])
        bt, L, D = x.size()
        x_cls = x[:, 0, :]  # cls_tokens  torch.Size([40, 768])
        x_cls_split = x_cls.view(B, T, D)  # torch.Size([4, 10, 768])
        shift_index = self.left_rotate_by_one(T)
        x_cls_shift = x_cls_split[:, shift_index, :].view(B * T, D)  # torch.Size([4, 10, 768])

        x_patch = x[:, 1:, :]
        #
        # x_fuse1 = torch.cat((x_cls.unsqueeze(1), x_patch_shift), dim=1)  # torch.Size([40, 163, 768])
        x_fuse_intra = self.Reconstuct_shift(x_cls_shift, x_patch)  # torch.Size([40, 163, 768])

        return x_fuse_intra

    def left_rotate_by_one(self, T):
        list_raw = [i for i in range(T)]
        list_new = list_raw[1:] + list_raw[:1]
        return list_new

    @staticmethod
    def shift(x, stride=1, n_div=4, bidirectional=True, padding='repeat'):
        B, L, D, l = x.size()
        fold = D // n_div  # number of fetures to be shifted
        if padding == 'zero':
            out = torch.zeros_like(x)
            if bidirectional:
                out[:, stride:, :fold] = x[:, :-stride, :fold]
                out[:, :-stride, fold:2 * fold] = x[:, stride:, fold:2 * fold]
                if n_div != 2:
                    out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
            else:  # 11
                out[:, stride:, :fold] = x[:, :-stride, :fold]
                out[:, :, fold:] = x[:, :, fold:]
        elif padding == 'repeat':
            out = x
            if bidirectional:
                out[:, stride:, :fold] = out[:, :-stride, :fold]
                out[:, :-stride, fold:2 * fold] = out[:, stride:, fold:2 * fold]
            else:
                out[:, stride:, :fold] = out[:, :-stride, :fold]
        return out