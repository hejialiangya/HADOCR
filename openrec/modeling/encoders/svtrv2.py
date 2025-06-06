import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, ones_, trunc_normal_, zeros_

# -----------------------------------------------------------
# 1) 公共功能模块：DropPath, Identity, Mlp, DropPath等
# -----------------------------------------------------------

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the naming differs to avoid increasing confusion.
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class Identity(nn.Module):
    def forward(self, x):
        return x

class Mlp(nn.Module):
    """MLP for token-mixer=Global/Local（输入为3D张量 [B, N, C]）"""
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# -----------------------------------------------------------
# 2) ConvBNLayer, ConvMixer, ConvMlp, Attention 等
# -----------------------------------------------------------

class ConvBNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=False,
        groups=1,
        act=nn.GELU,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


class ConvMixer(nn.Module):
    """用于 mixer='Conv' 时的混合操作（输入为4D张量 [B, C, H, W]）"""
    def __init__(
        self,
        dim,
        num_heads=8,
        local_k=[5, 5],
    ):
        super().__init__()
        # 这里可额外叠加一层Conv或其它操作使之“更强大”
        # 下述示例：在原本 3×3 conv 基础上，再增加一层可选DW Conv
        # 仅作示例，你可以按需扩展
        self.local_mixer = nn.Conv2d(dim, dim, 5, 1, 2, groups=num_heads)
        # 示例：再加一个1×1卷积来做额外的通道混合
        self.extra_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.extra_bn = nn.BatchNorm2d(dim)
        self.extra_act = nn.GELU()

    def forward(self, x, mask=None):
        x = self.local_mixer(x)
        # 新增的额外卷积操作
        y = self.extra_conv(x)
        y = self.extra_bn(y)
        y = self.extra_act(y)
        return x + y  # 残差连接，使之更具表达力


class ConvMlp(nn.Module):
    """用于 mixer='Conv' 时的 MLP（输入为4D张量 [B, C, H, W]）"""
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        groups=1,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=groups)
        self.act = act_layer()
        # 可叠加一次DWConv增强
        self.dw_conv = nn.Conv2d(hidden_features, hidden_features,
                                 kernel_size=3, padding=1,
                                 groups=hidden_features, bias=False)
        self.dw_bn = nn.BatchNorm2d(hidden_features)
        self.dw_act = act_layer()

        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dw_conv(x)
        x = self.dw_bn(x)
        x = self.dw_act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """用于 mixer='Global'/'Local' 的标准 Attention（输入为3D张量 [B, N, C]）"""
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 可选：加入相对位置编码、更多技巧以增强
        # 这里只做演示，若你不需要，可删除
        self.use_rel_pos = False
        if self.use_rel_pos:
            self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, 1000, 1000))

    def forward(self, x, mask=None):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = q @ k.transpose(-2, -1) * self.scale
        if mask is not None:
            attn += mask.unsqueeze(0)
        # 可选：如果使用相对位置编码
        if self.use_rel_pos:
            # 这里只是演示，还需根据序列长度 N 进行索引操作
            # attn = attn + self.rel_pos_bias[..., :N, :N]
            pass

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# -----------------------------------------------------------
# 3) 关键增强：ShapeAwareSE - 兼容 3D/4D 输入的“SE注意力”模块
# -----------------------------------------------------------

class ShapeAwareSE(nn.Module):
    """
    针对 SVTR 的 Block，因其可能传递 3D 或 4D 张量，这里做一个自适应维度的 SE。
      - 对 4D (B,C,H,W) 分支使用 2D SE
      - 对 3D (B,N,C) 分支使用 1D SE
    其核心思路：先进行全局池化(对应维度)，再经过一系列通道变换，最后用 sigmoid 加权到原输入上。
    """

    def __init__(self, channels, rd_ratio=0.25, act_layer=nn.GELU):
        super().__init__()
        self.channels = channels
        rd_channels = int(channels * rd_ratio)
        # 针对4D输入的SE
        self.pool2d = nn.AdaptiveAvgPool2d(1)
        self.conv1_2d = nn.Conv2d(channels, rd_channels, kernel_size=1, bias=False)
        self.bn1_2d = nn.BatchNorm2d(rd_channels)
        self.act2d = act_layer()
        # 深度可分离卷积(可选)
        self.dw_2d = nn.Conv2d(rd_channels, rd_channels, kernel_size=3,
                               padding=1, groups=rd_channels, bias=False)
        self.bn2_2d = nn.BatchNorm2d(rd_channels)
        self.conv2_2d = nn.Conv2d(rd_channels, channels, kernel_size=1, bias=True)

        # 针对3D输入的SE
        # 做法：先 B,N,C -> B,C,N -> 全局池化(N维) -> MLP -> B,C,N -> B,N,C
        self.pool1d = nn.AdaptiveAvgPool1d(1)
        self.conv1_1d = nn.Conv1d(channels, rd_channels, kernel_size=1, bias=False)
        self.bn1_1d = nn.BatchNorm1d(rd_channels)
        self.act1d = act_layer()
        self.dw_1d = nn.Conv1d(rd_channels, rd_channels, kernel_size=3,
                               padding=1, groups=rd_channels, bias=False)
        self.bn2_1d = nn.BatchNorm1d(rd_channels)
        self.conv2_1d = nn.Conv1d(rd_channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor):
        if x.dim() == 4:
            # B,C,H,W
            y = self.pool2d(x)                          # B,C,1,1
            y = self.conv1_2d(y)
            y = self.bn1_2d(y)
            y = self.act2d(y)
            y = self.dw_2d(y)
            y = self.bn2_2d(y)
            y = self.act2d(y)
            y = self.conv2_2d(y)                        # B,C,1,1
            y = torch.sigmoid(y)
            return x * y
        elif x.dim() == 3:
            # B,N,C
            x_t = x.transpose(1, 2)                     # B,C,N
            y = self.pool1d(x_t)                        # B,C,1
            y = self.conv1_1d(y)
            y = self.bn1_1d(y)
            y = self.act1d(y)
            y = self.dw_1d(y)
            y = self.bn2_1d(y)
            y = self.act1d(y)
            y = self.conv2_1d(y)                        # B,C,1
            y = torch.sigmoid(y)
            y = y.expand_as(x_t)                        # B,C,N
            out = x_t * y                               # B,C,N
            out = out.transpose(1, 2)                   # B,N,C
            return out
        else:
            # 其它维度不处理
            return x

# -----------------------------------------------------------
# 4) Block：在原先的 Attention / MLP 或 ConvMixer / ConvMlp 流程后面，增加 SE 注意力
# -----------------------------------------------------------

class FlattenTranspose(nn.Module):
    def forward(self, x, mask=None):
        return x.flatten(2).transpose(1, 2)


class Block(nn.Module):
    """
    Block 根据 mixer 不同而可能走 Attention+Mlp 或 ConvMixer+ConvMlp 分支。
    这里我们在最后增加 ShapeAwareSE，使得整体更有表达力。
    """

    def __init__(
        self,
        dim,
        num_heads,
        mixer='Global',
        local_k=[7, 11],
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        eps=1e-6,
        use_se=True,              # 新增：是否使用 ShapeAwareSE
        se_ratio=0.25,           # 新增：SE通道缩放比率
    ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mixer_type = mixer
        if mixer == 'Global' or mixer == 'Local':
            self.norm1 = norm_layer(dim, eps=eps)
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
            self.norm2 = norm_layer(dim, eps=eps)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
        elif mixer == 'Conv':
            self.norm1 = nn.BatchNorm2d(dim)
            self.mixer = ConvMixer(dim, num_heads=num_heads, local_k=local_k)
            self.norm2 = nn.BatchNorm2d(dim)
            self.mlp = ConvMlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
        else:
            raise TypeError('The mixer must be one of [Global, Local, Conv]')

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()

        # 新增：可选的 ShapeAwareSE
        self.use_se = use_se
        if self.use_se:
            self.se = ShapeAwareSE(channels=dim, rd_ratio=se_ratio, act_layer=act_layer)
        else:
            self.se = Identity()

    def forward(self, x, mask=None):
        if self.mixer_type in ['Global', 'Local']:
            # x shape = [B, N, C]
            x = self.norm1(x + self.drop_path(self.mixer(x, mask=mask)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
            # 最后加 SE
            x = self.se(x)
        else:
            # mixer_type == 'Conv'
            # x shape = [B, C, H, W]
            x = self.norm1(x + self.drop_path(self.mixer(x, mask=mask)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
            # 最后加 SE
            x = self.se(x)
        return x


# -----------------------------------------------------------
# 5) SVTRStage 和 主体网络 SVTRv2
# -----------------------------------------------------------

class SVTRStage(nn.Module):
    def __init__(self,
                 feat_maxSize=[16, 128],
                 dim=64,
                 out_dim=256,
                 depth=3,
                 mixer=['Local'] * 3,
                 local_k=[7, 11],
                 sub_k=[2, 1],
                 num_heads=2,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=[0.1] * 3,
                 norm_layer=nn.LayerNorm,
                 act=nn.GELU,
                 eps=1e-6,
                 downsample=None,
                 # 新增：可在每个Stage配置是否用SE
                 use_se=True,
                 se_ratio=0.25,
                 **kwargs):
        super().__init__()
        self.dim = dim

        # 判断本 Stage 里有多少个 mixer='Conv'
        conv_block_num = sum([1 if mix == 'Conv' else 0 for mix in mixer])
        # 如果全部是Conv，可简化处理
        if conv_block_num == depth:
            self.mask = None
            conv_block_num = 0
            if downsample:
                self.sub_norm = nn.BatchNorm2d(out_dim, eps=eps)
        else:
            # 如果有 Local，需要做 Mask
            if 'Local' in mixer:
                mask = self.get_max2d_mask(feat_maxSize[0], feat_maxSize[1], local_k)
                self.register_buffer('mask', mask)
            else:
                self.mask = None
            if downsample:
                self.sub_norm = norm_layer(out_dim, eps=eps)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                Block(
                    dim=dim,
                    num_heads=num_heads,
                    mixer=mixer[i],
                    local_k=local_k,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                    norm_layer=norm_layer,
                    act_layer=act,
                    eps=eps,
                    use_se=use_se,       # 传入可选SE
                    se_ratio=se_ratio,
                ))
            if i == conv_block_num - 1:
                self.blocks.append(FlattenTranspose())

        if downsample:
            self.downsample = nn.Conv2d(dim,
                                        out_dim,
                                        kernel_size=3,
                                        stride=sub_k,
                                        padding=1)
        else:
            self.downsample = None

    def get_max2d_mask(self, H, W, local_k):
        hk, wk = local_k
        mask = torch.ones(H * W, H + hk - 1, W + wk - 1,
                          dtype=torch.float32,
                          requires_grad=False)
        for h in range(0, H):
            for w in range(0, W):
                mask[h * W + w, h:h + hk, w:w + wk] = 0.0
        mask = mask[:, hk // 2:H + hk // 2, wk // 2:W + wk // 2]
        mask[mask >= 1] = -np.inf
        return mask.reshape(H, W, H, W)

    def get_2d_mask(self, H1, W1):
        if H1 == self.mask.shape[0] and W1 == self.mask.shape[1]:
            return self.mask.flatten(0, 1).flatten(1, 2).unsqueeze(0)
        h_slice = H1 // 2
        offet_h = H1 - 2 * h_slice
        w_slice = W1 // 2
        offet_w = W1 - 2 * w_slice
        mask1 = self.mask[:h_slice + offet_h, :w_slice, :H1, :W1]
        mask2 = self.mask[:h_slice + offet_h, -w_slice:, :H1, -W1:]
        mask3 = self.mask[-h_slice:, :(w_slice + offet_w), -H1:, :W1]
        mask4 = self.mask[-h_slice:, -(w_slice + offet_w):, -H1:, -W1:]

        mask_top = torch.concat([mask1, mask2], 1)
        mask_bott = torch.concat([mask3, mask4], 1)
        mask = torch.concat([mask_top.flatten(2), mask_bott.flatten(2)], 0)
        return mask.flatten(0, 1).unsqueeze(0)

    def forward(self, x, sz=None):
        if self.mask is not None:
            mask = self.get_2d_mask(sz[0], sz[1])
        else:
            mask = self.mask

        for blk in self.blocks:
            x = blk(x, mask=mask)

        if self.downsample is not None:
            if x.dim() == 3:
                # [B, N, C] => [B, C, H, W]
                x = x.transpose(1, 2).reshape(-1, self.dim, sz[0], sz[1])
                x = self.downsample(x)
                sz = x.shape[2:]
                x = x.flatten(2).transpose(1, 2)
            else:
                # 4D -> DownSample
                x = self.downsample(x)
                sz = x.shape[2:]
            x = self.sub_norm(x)
        return x, sz


class POPatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    def __init__(self,
                 in_channels=3,
                 feat_max_size=[8, 32],
                 embed_dim=768,
                 use_pos_embed=False,
                 flatten=False):
        super().__init__()
        self.patch_embed = nn.Sequential(
            ConvBNLayer(
                in_channels=in_channels,
                out_channels=embed_dim // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                act=nn.GELU,
                bias=None,
            ),
            ConvBNLayer(
                in_channels=embed_dim // 2,
                out_channels=embed_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                act=nn.GELU,
                bias=None,
            ),
        )
        self.use_pos_embed = use_pos_embed
        self.flatten = flatten
        if use_pos_embed:
            pos_embed = torch.zeros(
                [1, feat_max_size[0] * feat_max_size[1], embed_dim],
                dtype=torch.float32)
            trunc_normal_(pos_embed, mean=0, std=0.02)
            self.pos_embed = nn.Parameter(
                pos_embed.transpose(1, 2).reshape(
                    1, embed_dim, feat_max_size[0], feat_max_size[1]
                ),
                requires_grad=True,
            )

    def forward(self, x):
        x = self.patch_embed(x)
        sz = x.shape[2:]
        if self.use_pos_embed:
            x = x + self.pos_embed[:, :, :sz[0], :sz[1]]
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return x, sz


class SVTRv2(nn.Module):
    """
    增强版 SVTRv2：
      - 在每个 Block 的末尾增加 “ShapeAwareSE” 可选通道注意力
      - 保持原先输入输出形状和整体流程不变
    """
    def __init__(
        self,
        max_sz=[32, 128],
        in_channels=3,
        out_channels=192,
        depths=[3, 6, 3],
        dims=[64, 128, 256],
        mixer=[
            ['Local'] * 3,
            ['Local'] * 3 + ['Global'] * 3,
            ['Global'] * 3
        ],
        use_pos_embed=True,
        local_k=[[7, 11], [7, 11], [-1, -1]],
        sub_k=[[1, 1], [2, 1], [1, 1]],
        num_heads=[2, 4, 8],
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        last_drop=0.1,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        act=nn.GELU,
        last_stage=False,
        eps=1e-6,
        # 新增：可在全局指定是否用SE、SE缩放比
        use_se=True,
        se_ratio=0.25,
        **kwargs
    ):
        super().__init__()
        num_stages = len(depths)
        self.num_features = dims[-1]

        feat_max_size = [max_sz[0] // 4, max_sz[1] // 4]
        self.pope = POPatchEmbed(
            in_channels=in_channels,
            feat_max_size=feat_max_size,
            embed_dim=dims[0],
            use_pos_embed=use_pos_embed,
            flatten=mixer[0][0] != 'Conv',
        )

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, sum(depths))

        self.stages = nn.ModuleList()
        for i_stage in range(num_stages):
            stage = SVTRStage(
                feat_maxSize=feat_max_size,
                dim=dims[i_stage],
                out_dim=dims[i_stage + 1] if i_stage < num_stages - 1 else 0,
                depth=depths[i_stage],
                mixer=mixer[i_stage],
                local_k=local_k[i_stage],
                sub_k=sub_k[i_stage],
                num_heads=num_heads[i_stage],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                norm_layer=norm_layer,
                act=act,
                downsample=False if i_stage == num_stages - 1 else True,
                eps=eps,
                use_se=use_se,       # 将全局的 use_se / se_ratio 传入
                se_ratio=se_ratio,
            )
            self.stages.append(stage)
            feat_max_size = [
                feat_max_size[0] // sub_k[i_stage][0],
                feat_max_size[1] // sub_k[i_stage][1],
            ]

        self.out_channels = self.num_features
        self.last_stage = last_stage
        if last_stage:
            self.out_channels = out_channels
            self.last_conv = nn.Linear(self.num_features, self.out_channels, bias=False)
            self.hardswish = nn.Hardswish()
            self.dropout = nn.Dropout(p=last_drop)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        if isinstance(m, nn.Conv2d):
            kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'patch_embed', 'downsample', 'pos_embed'}

    def forward(self, x):
        """
        Args:
            x (Tensor): shape=[B, in_channels, H, W]
        Returns:
            (Tensor): 如果 last_stage=False，则输出 [B, N, C] 或 [B, C, H, W] 视mixer而定
                      如果 last_stage=True，则输出 [B, out_channels]
        """
        x, sz = self.pope(x)

        for stage in self.stages:
            x, sz = stage(x, sz)

        if self.last_stage:
            x = x.reshape(-1, sz[0], sz[1], self.num_features)
            # global pooling along spatial dims
            x = x.mean(1)
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x)

        return x
