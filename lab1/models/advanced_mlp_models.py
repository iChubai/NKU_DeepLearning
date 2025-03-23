import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBlock(nn.Module):
    """标准MLP块，用于MLP-Mixer和ResMLP"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class MixerBlock(nn.Module):
    """MLP-Mixer的基本块"""
    def __init__(self, tokens_dim, channels_dim, token_hidden_dim, channel_hidden_dim):
        super().__init__()
        # 修改token_mix实现方式
        self.norm1 = nn.LayerNorm(channels_dim)
        self.token_mlp = MLPBlock(tokens_dim, token_hidden_dim)
        
        # 修改channel_mix实现方式
        self.norm2 = nn.LayerNorm(channels_dim)
        self.channel_mlp = MLPBlock(channels_dim, channel_hidden_dim)
        
    def forward(self, x):
        # x: [batch_size, tokens_dim, channels_dim]
        # Token-mixing: 先LayerNorm，再转置操作
        residual = x
        x_norm = self.norm1(x)
        x_t = x_norm.transpose(1, 2)  # [batch_size, channels_dim, tokens_dim]
        x_t = self.token_mlp(x_t)     # 在token维度上进行MLP操作
        x = residual + x_t.transpose(1, 2)  # 转置回来并应用残差连接
        
        # Channel-mixing
        residual = x
        x = self.norm2(x)
        x = self.channel_mlp(x)
        x = residual + x
        
        return x

class MLPMixer(nn.Module):
    """MLP-Mixer模型实现"""
    def __init__(self, input_dim=784, image_size=28, patch_size=4, num_classes=10, 
                 num_blocks=4, hidden_dim=256, token_hidden_dim=256, channel_hidden_dim=512):
        super().__init__()
        
        # 计算每个维度的patch数量和总patch数量
        num_patches_per_dim = image_size // patch_size
        num_patches = num_patches_per_dim ** 2
        patch_dim = (patch_size ** 2)
        
        self.patch_size = patch_size
        self.num_patches_per_dim = num_patches_per_dim
        
        # Patch嵌入层
        self.patch_embed = nn.Linear(patch_dim, hidden_dim)
        
        # Mixer层
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(
                tokens_dim=num_patches,
                channels_dim=hidden_dim,
                token_hidden_dim=token_hidden_dim,
                channel_hidden_dim=channel_hidden_dim
            )
            for _ in range(num_blocks)
        ])
        
        # 规范化和分类头
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x: [batch_size, channels, height, width]
        batch_size = x.shape[0]
        
        # 将图像重塑为patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, 1, self.num_patches_per_dim, self.num_patches_per_dim, self.patch_size * self.patch_size)
        x = x.view(batch_size, self.num_patches_per_dim * self.num_patches_per_dim, -1)
        
        # 将patches线性投影到隐藏维度
        x = self.patch_embed(x)
        
        # 应用Mixer块
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        
        # 全局平均池化
        x = self.norm(x).mean(dim=1)
        
        # 分类
        x = self.classifier(x)
        return x

class ResMLPBlock(nn.Module):
    """ResMLP块"""
    def __init__(self, dim, hidden_dim, num_patches=49):
        super().__init__()
        self.affine1 = nn.Parameter(torch.ones(1, 1, dim))
        # 修改spatial_mlp实现，它应该在patches维度上操作
        self.spatial_mlp = nn.Linear(num_patches, num_patches)
        self.affine2 = nn.Parameter(torch.ones(1, 1, dim))
        self.channel_mlp = MLPBlock(dim, hidden_dim)
        
    def forward(self, x):
        # x: [batch_size, num_patches, dim]
        # 空间MLP (cross-patch)
        residual = x
        x = x * self.affine1
        # 转置为[batch_size, dim, num_patches]以使spatial_mlp作用于patches维度
        x = x.transpose(1, 2)
        # 在patches维度上应用MLP
        x = self.spatial_mlp(x)
        # 转置回原始形状
        x = x.transpose(1, 2)
        x = residual + x
        
        # 通道MLP (in-patch)
        residual = x
        x = x * self.affine2
        x = self.channel_mlp(x)
        x = residual + x
        
        return x

class ResMLP(nn.Module):
    """ResMLP模型实现"""
    def __init__(self, input_dim=784, image_size=28, patch_size=4, num_classes=10, 
                 num_blocks=4, hidden_dim=384, mlp_hidden_dim=768):
        super().__init__()
        
        # 计算每个维度的patch数量和总patch数量
        num_patches_per_dim = image_size // patch_size
        num_patches = num_patches_per_dim ** 2
        patch_dim = (patch_size ** 2)
        
        self.patch_size = patch_size
        self.num_patches_per_dim = num_patches_per_dim
        
        # Patch嵌入层
        self.patch_embed = nn.Linear(patch_dim, hidden_dim)
        
        # ResMLP块，传入num_patches参数
        self.blocks = nn.ModuleList([
            ResMLPBlock(dim=hidden_dim, hidden_dim=mlp_hidden_dim, num_patches=num_patches)
            for _ in range(num_blocks)
        ])
        
        # 规范化和分类头
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x: [batch_size, channels, height, width]
        batch_size = x.shape[0]
        
        # 将图像重塑为patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, 1, self.num_patches_per_dim, self.num_patches_per_dim, self.patch_size * self.patch_size)
        x = x.view(batch_size, self.num_patches_per_dim * self.num_patches_per_dim, -1)
        
        # 将patches线性投影到隐藏维度
        x = self.patch_embed(x)
        
        # 应用ResMLP块
        for block in self.blocks:
            x = block(x)
        
        # 全局平均池化
        x = self.norm(x).mean(dim=1)
        
        # 分类
        x = self.classifier(x)
        return x

class PermutatorBlock(nn.Module):
    """Vision Permutator块"""
    def __init__(self, dim, hidden_dim, segment_dim=8):
        super().__init__()
        self.segment_dim = segment_dim
        
        # 确保dim可以被segment_dim整除
        assert dim % segment_dim == 0, f"维度 {dim} 必须能被segment_dim {segment_dim} 整除"
        
        # 三个维度的Permutator
        self.norm1 = nn.LayerNorm(dim)
        self.h_perm = nn.Linear(segment_dim, segment_dim)
        
        self.norm2 = nn.LayerNorm(dim)
        self.w_perm = nn.Linear(segment_dim, segment_dim)
        
        self.norm3 = nn.LayerNorm(dim)
        self.c_mlp = MLPBlock(dim, hidden_dim)
    
    def forward(self, x):
        # x: [batch_size, h*w, dim]
        batch_size, seq_len, dim = x.shape
        h = w = int(seq_len ** 0.5)
        
        # 高度维度Permutator
        residual = x
        x = self.norm1(x)
        x = x.view(batch_size, h, w, dim)
        
        # 将dim重塑为segment_dim组
        x = x.view(batch_size, h, w, dim // self.segment_dim, self.segment_dim)
        x = x.permute(0, 3, 2, 1, 4).contiguous()  # [batch, dim//seg_dim, w, h, seg_dim]
        x = self.h_perm(x)
        x = x.permute(0, 3, 2, 1, 4).contiguous()  # [batch, h, w, dim//seg_dim, seg_dim]
        x = x.view(batch_size, h, w, dim)
        x = x.view(batch_size, h * w, dim)
        x = residual + x
        
        # 宽度维度Permutator
        residual = x
        x = self.norm2(x)
        x = x.view(batch_size, h, w, dim)
        
        # 将dim重塑为segment_dim组
        x = x.view(batch_size, h, w, dim // self.segment_dim, self.segment_dim)
        x = x.permute(0, 3, 1, 2, 4).contiguous()  # [batch, dim//seg_dim, h, w, seg_dim]
        x = self.w_perm(x)
        x = x.permute(0, 2, 3, 1, 4).contiguous()  # [batch, h, w, dim//seg_dim, seg_dim]
        x = x.view(batch_size, h, w, dim)
        x = x.view(batch_size, h * w, dim)
        x = residual + x
        
        # 通道维度MLP
        residual = x
        x = self.norm3(x)
        x = self.c_mlp(x)
        x = residual + x
        
        return x

class VisionPermutator(nn.Module):
    """Vision Permutator模型实现"""
    def __init__(self, input_dim=784, image_size=28, patch_size=4, num_classes=10, 
                 num_blocks=4, hidden_dim=384, mlp_hidden_dim=768, segment_dim=8):
        super().__init__()
        
        # 计算每个维度的patch数量和总patch数量
        num_patches_per_dim = image_size // patch_size
        num_patches = num_patches_per_dim ** 2
        patch_dim = (patch_size ** 2)
        
        self.patch_size = patch_size
        self.num_patches_per_dim = num_patches_per_dim
        
        # Patch嵌入层
        self.patch_embed = nn.Linear(patch_dim, hidden_dim)
        
        # Vision Permutator块
        self.blocks = nn.ModuleList([
            PermutatorBlock(dim=hidden_dim, hidden_dim=mlp_hidden_dim, segment_dim=segment_dim)
            for _ in range(num_blocks)
        ])
        
        # 规范化和分类头
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x: [batch_size, channels, height, width]
        batch_size = x.shape[0]
        
        # 将图像重塑为patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, 1, self.num_patches_per_dim, self.num_patches_per_dim, self.patch_size * self.patch_size)
        x = x.view(batch_size, self.num_patches_per_dim * self.num_patches_per_dim, -1)
        
        # 将patches线性投影到隐藏维度
        x = self.patch_embed(x)
        
        # 应用Vision Permutator块
        for block in self.blocks:
            x = block(x)
        
        # 全局平均池化
        x = self.norm(x).mean(dim=1)
        
        # 分类
        x = self.classifier(x)
        return x 