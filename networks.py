#2 G-D
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Helper Layers ---
class WSConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.scale = (gain / (in_channels * (kernel_size ** 3))) ** 0.5
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, -1, 1, 1, 1)

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8
    
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

class MinibatchStdDev(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        batch_size, _, depth, height, width = x.shape
        y = x - torch.mean(x, dim=0, keepdim=True)
        y = torch.mean(y ** 2, dim=0)
        y = torch.sqrt(y + 1e-8)
        y = torch.mean(y).view(1, 1, 1, 1, 1)
        y = y.repeat(batch_size, 1, depth, height, width)
        return torch.cat([x, y], dim=1)

# --- Generator Blocks ---
class G_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = WSConv3d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = WSConv3d(out_channels, out_channels, 3, 1, 1)
        self.pixel_norm = PixelNorm()
        self.lrelu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        x = self.lrelu(self.pixel_norm(self.conv1(x)))
        x = self.lrelu(self.pixel_norm(self.conv2(x)))
        return x

# --- Generator for Baseline (Standard PGGAN) ---
class Generator_Baseline(nn.Module):
    """Generator for baseline images: noise → baseline"""
    def __init__(self, latent_size=256, num_channels=1, fmap_base=4096, fmap_max=256, max_res_log2=6):
        super().__init__()
        
        def nf(stage):
            return min(int(fmap_base / (2.0 ** stage)), fmap_max)
        
        self.initial_block = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose3d(latent_size, nf(0), 4, 1, 0),
            nn.LeakyReLU(0.2), PixelNorm(),
            WSConv3d(nf(0), nf(0), 3, 1, 1),
            nn.LeakyReLU(0.2), PixelNorm(),
        )
        
        self.prog_blocks = nn.ModuleList()
        self.to_rgb_layers = nn.ModuleList()
        
        self.to_rgb_layers.append(WSConv3d(nf(0), num_channels, 1, 1, 0))
        
        for i in range(1, max_res_log2 - 1):
            in_ch, out_ch = nf(i-1), nf(i)
            self.prog_blocks.append(G_Block(in_ch, out_ch))
            self.to_rgb_layers.append(WSConv3d(out_ch, num_channels, 1, 1, 0))
    
    def forward(self, z, level, alpha):
        level = int(level)
        x = self.initial_block(z.view(z.shape[0], -1, 1, 1, 1))
        
        if level == 0:
            return self.to_rgb_layers[0](x)
        
        for i in range(level):
            x_prev = x
            x = self.prog_blocks[i](x)
        
        out_stable = self.to_rgb_layers[level](x)
        
        if alpha < 1.0:
            out_faded = F.interpolate(self.to_rgb_layers[level-1](x_prev), scale_factor=2, mode='trilinear', align_corners=False)
            return (1 - alpha) * out_faded + alpha * out_stable
        
        return out_stable

# --- Generator for Follow-up (Conditioned on Baseline) ---
class Generator_Followup(nn.Module):
    """Generator for follow-up: baseline + noise → follow-up"""
    def __init__(self, latent_size=256, num_channels=1, fmap_base=4096, fmap_max=256, max_res_log2=6):
        super().__init__()
        
        def nf(stage):
            return min(int(fmap_base / (2.0 ** stage)), fmap_max)
        
        # Initial block takes baseline (1 channel) + noise features
        self.initial_block = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose3d(latent_size, nf(0), 4, 1, 0),
            nn.LeakyReLU(0.2), PixelNorm(),
        )
        
        # Merge baseline with noise features at 4x4x4
        self.merge_initial = nn.Sequential(
            WSConv3d(nf(0) + num_channels, nf(0), 3, 1, 1),
            nn.LeakyReLU(0.2), PixelNorm(),
            WSConv3d(nf(0), nf(0), 3, 1, 1),
            nn.LeakyReLU(0.2), PixelNorm(),
        )
        
        self.prog_blocks = nn.ModuleList()
        self.merge_blocks = nn.ModuleList()
        self.to_rgb_layers = nn.ModuleList()
        
        self.to_rgb_layers.append(WSConv3d(nf(0), num_channels, 1, 1, 0))
        
        for i in range(1, max_res_log2 - 1):
            in_ch, out_ch = nf(i-1), nf(i)
            # Progressive upsampling block
            self.prog_blocks.append(G_Block(in_ch, out_ch))
            # Merge block to combine with baseline at this resolution
            merge_block = nn.Sequential(
                WSConv3d(out_ch + num_channels, out_ch, 3, 1, 1),
                nn.LeakyReLU(0.2), PixelNorm(),
                WSConv3d(out_ch, out_ch, 3, 1, 1),
                nn.LeakyReLU(0.2), PixelNorm(),
            )
            self.merge_blocks.append(merge_block)
            self.to_rgb_layers.append(WSConv3d(out_ch, num_channels, 1, 1, 0))
    
    def forward(self, z, baseline, level, alpha):
        """
        Args:
            z: noise [batch, latent_size]
            baseline: synthetic baseline [batch, 1, res, res, res]
            level: current resolution level
            alpha: fade-in alpha
        """
        level = int(level)
        current_res = 4 * (2 ** level)
        
        # Generate noise features
        x = self.initial_block(z.view(z.shape[0], -1, 1, 1, 1))
        
        # Downsample baseline to 4x4x4 and merge
        baseline_4x4 = F.interpolate(baseline, size=(4, 4, 4), mode='trilinear', align_corners=False)
        x = torch.cat([x, baseline_4x4], dim=1)
        x = self.merge_initial(x)
        
        if level == 0:
            return self.to_rgb_layers[0](x)
        
        # Progressive upsampling with baseline conditioning
        for i in range(level):
            x_prev = x
            x = self.prog_blocks[i](x)
            
            # Upsample baseline to current resolution and merge
            current_size = x.shape[2]
            baseline_upsampled = F.interpolate(baseline, size=(current_size, current_size, current_size), 
                                              mode='trilinear', align_corners=False)
            x = torch.cat([x, baseline_upsampled], dim=1)
            x = self.merge_blocks[i](x)
        
        out_stable = self.to_rgb_layers[level](x)
        
        if alpha < 1.0:
            out_faded = F.interpolate(self.to_rgb_layers[level-1](x_prev), scale_factor=2, mode='trilinear', align_corners=False)
            return (1 - alpha) * out_faded + alpha * out_stable
        
        return out_stable

# --- Discriminator Blocks ---
class D_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = WSConv3d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = WSConv3d(in_channels, out_channels, 3, 1, 1)
        self.downsample = nn.AvgPool3d(2)
        self.lrelu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        y = self.lrelu(self.conv1(x))
        y = self.lrelu(self.conv2(y))
        y = self.downsample(y)
        return y

# --- Discriminator (Same architecture for both baseline and follow-up) ---
class Discriminator(nn.Module):
    def __init__(self, num_channels=1, fmap_base=4096, fmap_max=256, max_res_log2=6):
        super().__init__()
        
        def nf(stage):
            return min(int(fmap_base / (2.0 ** stage)), fmap_max)

        self.prog_blocks = nn.ModuleList()
        self.from_rgb_layers = nn.ModuleList()

        for i in range(max_res_log2 - 2, -1, -1):
            in_ch = nf(i + 1)
            out_ch = nf(i)
            self.prog_blocks.append(D_Block(in_ch, out_ch))
            self.from_rgb_layers.append(nn.Sequential(WSConv3d(num_channels, in_ch, 1, 1, 0), nn.LeakyReLU(0.2)))
        
        self.from_rgb_layers.append(nn.Sequential(WSConv3d(num_channels, nf(0), 1, 1, 0), nn.LeakyReLU(0.2)))
        
        temp_list = list(self.from_rgb_layers)
        temp_list.reverse()
        self.from_rgb_layers = nn.ModuleList(temp_list)

        self.final_block = nn.Sequential(
            MinibatchStdDev(),
            WSConv3d(nf(0) + 1, nf(0), 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(nf(0) * (4**3), nf(0)), nn.LeakyReLU(0.2),
            nn.Linear(nf(0), 1)
        )
        
        self.max_res_log2 = max_res_log2
        
    def forward(self, x, level, alpha):
        level = int(level)
        
        y = self.from_rgb_layers[level](x)

        if level > 0 and alpha < 1.0:
            y_main_path = self.prog_blocks[len(self.prog_blocks) - level](y)
            y_prev_path = F.avg_pool3d(x, 2)
            y_prev_path = self.from_rgb_layers[level - 1](y_prev_path)
            y = (1.0 - alpha) * y_prev_path + alpha * y_main_path
            
            for i in range(len(self.prog_blocks) - level + 1, len(self.prog_blocks)):
                y = self.prog_blocks[i](y)
        else:
            start_block_idx = len(self.prog_blocks) - level
            for i in range(start_block_idx, len(self.prog_blocks)):
                y = self.prog_blocks[i](y)
        
        score = self.final_block(y)
        return score, None
# #D-code
# # networks.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# class WSConv3d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, gain=2):
#         super().__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
#         self.scale = (gain / (in_channels * (kernel_size ** 3))) ** 0.5
#         self.bias = nn.Parameter(torch.zeros(out_channels))
#         nn.init.normal_(self.conv.weight)
#         nn.init.zeros_(self.bias)
#     def forward(self, x):
#         return self.conv(x * self.scale) + self.bias.view(1, -1, 1, 1, 1)

# class PixelNorm(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.epsilon = 1e-8
#     def forward(self, x):
#         return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

# class MinibatchStdDev(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x):
#         batch_size, _, depth, height, width = x.shape
#         y = x - torch.mean(x, dim=0, keepdim=True)
#         y = torch.mean(y ** 2, dim=0)
#         y = torch.sqrt(y + 1e-8)
#         y = torch.mean(y).view(1, 1, 1, 1, 1)
#         y = y.repeat(batch_size, 1, depth, height, width)
#         return torch.cat([x, y], dim=1)

# class G_Block(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = WSConv3d(in_channels, out_channels, 3, 1, 1)
#         self.conv2 = WSConv3d(out_channels, out_channels, 3, 1, 1)
#         self.pixel_norm = PixelNorm()
#         self.lrelu = nn.LeakyReLU(0.2)
#     def forward(self, x):
#         x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
#         x = self.lrelu(self.pixel_norm(self.conv1(x)))
#         x = self.lrelu(self.pixel_norm(self.conv2(x)))
#         return x

# class Generator(nn.Module):
#     def __init__(self, latent_size=256, num_channels=2, fmap_base=4096, fmap_max=256, max_res_log2=6):
#         super().__init__()
#         def nf(stage):
#             return min(int(fmap_base / (2.0 ** stage)), fmap_max)
#         self.initial_block = nn.Sequential(
#             PixelNorm(),
#             nn.ConvTranspose3d(latent_size, nf(0), 4, 1, 0),
#             nn.LeakyReLU(0.2), PixelNorm(),
#             WSConv3d(nf(0), nf(0), 3, 1, 1),
#             nn.LeakyReLU(0.2), PixelNorm(),
#         )
#         self.prog_blocks = nn.ModuleList()
#         self.to_rgb_layers = nn.ModuleList()
#         self.to_rgb_layers.append(WSConv3d(nf(0), num_channels, 1, 1, 0))
#         for i in range(1, max_res_log2 - 1):
#             in_ch, out_ch = nf(i-1), nf(i)
#             self.prog_blocks.append(G_Block(in_ch, out_ch))
#             self.to_rgb_layers.append(WSConv3d(out_ch, num_channels, 1, 1, 0))
#     def forward(self, x, level, alpha):
#         level = int(level)
#         x = self.initial_block(x.view(x.shape[0], -1, 1, 1, 1))
#         if level == 0:
#             return self.to_rgb_layers[0](x)
#         for i in range(level):
#             x_prev = x
#             x = self.prog_blocks[i](x)
#         out_stable = self.to_rgb_layers[level](x)
#         if alpha < 1.0:
#             out_faded = F.interpolate(self.to_rgb_layers[level-1](x_prev), scale_factor=2)
#             return (1 - alpha) * out_faded + alpha * out_stable
#         return out_stable

# class D_Block(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = WSConv3d(in_channels, in_channels, 3, 1, 1)
#         self.conv2 = WSConv3d(in_channels, out_channels, 3, 1, 1)
#         self.downsample = nn.AvgPool3d(2)
#         self.lrelu = nn.LeakyReLU(0.2)
#     def forward(self, x):
#         y = self.lrelu(self.conv1(x))
#         y = self.lrelu(self.conv2(y))
#         y = self.downsample(y)
#         return y

# class Discriminator(nn.Module):
#     def __init__(self, num_channels=2, fmap_base=4096, fmap_max=256, max_res_log2=6):
#         super().__init__()
#         def nf(stage):
#             return min(int(fmap_base / (2.0 ** stage)), fmap_max)
#         self.prog_blocks = nn.ModuleList()
#         self.from_rgb_layers = nn.ModuleList()
#         for i in range(max_res_log2 - 1):
#             in_ch = nf(i)
#             out_ch = nf(i-1) if i > 0 else nf(0)
#             self.from_rgb_layers.append(nn.Sequential(WSConv3d(num_channels, in_ch, 1, 1, 0), nn.LeakyReLU(0.2)))
#             self.prog_blocks.append(D_Block(in_ch, out_ch))
#         self.final_block = nn.Sequential(
#             MinibatchStdDev(),
#             WSConv3d(nf(0) + 1, nf(0), 3, 1, 1), nn.LeakyReLU(0.2),
#             nn.Flatten(),
#             nn.Linear(nf(0) * (4**3), nf(0)), nn.LeakyReLU(0.2),
#             nn.Linear(nf(0), 1)
#         )
#     def forward(self, x, level, alpha):
#         level = int(level)
#         y = self.from_rgb_layers[level](x)
#         if level > 0 and alpha < 1.0:
#             y_prev = F.avg_pool3d(x, 2)
#             y_prev = self.from_rgb_layers[level - 1](y_prev)
#             y = self.prog_blocks[level-1](y)
#             y = (1.0 - alpha) * y_prev + alpha * y
#             for i in range(level - 1, 0, -1):
#                 y = self.prog_blocks[i-1](y)
#         else:
#             for i in range(level, 0, -1):
#                 y = self.prog_blocks[i-1](y)
#         score = self.final_block(y)
#         return score, None
#H-code
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# # --- Helper Layers (Unchanged) ---
# class WSConv3d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, gain=2):
#         super().__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
#         self.scale = (gain / (in_channels * (kernel_size ** 3))) ** 0.5
#         self.bias = nn.Parameter(torch.zeros(out_channels))
#         nn.init.normal_(self.conv.weight)
#         nn.init.zeros_(self.bias)
#     def forward(self, x):
#         return self.conv(x * self.scale) + self.bias.view(1, -1, 1, 1, 1)

# class PixelNorm(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.epsilon = 1e-8
#     def forward(self, x):
#         return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

# class MinibatchStdDev(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x):
#         batch_size, _, depth, height, width = x.shape
#         y = x - torch.mean(x, dim=0, keepdim=True)
#         y = torch.mean(y ** 2, dim=0)
#         y = torch.sqrt(y + 1e-8)
#         y = torch.mean(y).view(1, 1, 1, 1, 1)
#         y = y.repeat(batch_size, 1, depth, height, width)
#         return torch.cat([x, y], dim=1)

# # --- Generator Blocks (Unchanged) ---
# class G_Block(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = WSConv3d(in_channels, out_channels, 3, 1, 1)
#         self.conv2 = WSConv3d(out_channels, out_channels, 3, 1, 1)
#         self.pixel_norm = PixelNorm()
#         self.lrelu = nn.LeakyReLU(0.2)
#     def forward(self, x):
#         x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
#         x = self.lrelu(self.pixel_norm(self.conv1(x)))
#         x = self.lrelu(self.pixel_norm(self.conv2(x)))
#         return x

# class Generator(nn.Module):
#     # CHANGED: Default num_channels is now 2
#     def __init__(self, latent_size=256, num_channels=2, fmap_base=4096, fmap_max=256, max_res_log2=6):
#         super().__init__()
#         def nf(stage):
#             return min(int(fmap_base / (2.0 ** stage)), fmap_max)
#         self.initial_block = nn.Sequential(
#             PixelNorm(),
#             nn.ConvTranspose3d(latent_size, nf(0), 4, 1, 0),
#             nn.LeakyReLU(0.2), PixelNorm(),
#             WSConv3d(nf(0), nf(0), 3, 1, 1),
#             nn.LeakyReLU(0.2), PixelNorm(),
#         )
#         self.prog_blocks = nn.ModuleList()
#         self.to_rgb_layers = nn.ModuleList()
#         self.to_rgb_layers.append(WSConv3d(nf(0), num_channels, 1, 1, 0))
#         for i in range(1, max_res_log2 - 1):
#             in_ch, out_ch = nf(i-1), nf(i)
#             self.prog_blocks.append(G_Block(in_ch, out_ch))
#             self.to_rgb_layers.append(WSConv3d(out_ch, num_channels, 1, 1, 0))
#     def forward(self, x, level, alpha):
#         level = int(level)
#         x = self.initial_block(x.view(x.shape[0], -1, 1, 1, 1))
#         if level == 0:
#             return self.to_rgb_layers[0](x)
#         for i in range(level):
#             x_prev = x
#             x = self.prog_blocks[i](x)
#         out_stable = self.to_rgb_layers[level](x)
#         if alpha < 1.0:
#             out_faded = F.interpolate(self.to_rgb_layers[level-1](x_prev), scale_factor=2)
#             return (1 - alpha) * out_faded + alpha * out_stable
#         return out_stable

# # --- Discriminator ---
# class D_Block(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = WSConv3d(in_channels, in_channels, 3, 1, 1)
#         self.conv2 = WSConv3d(in_channels, out_channels, 3, 1, 1)
#         self.downsample = nn.AvgPool3d(2)
#         self.lrelu = nn.LeakyReLU(0.2)
#     def forward(self, x):
#         y = self.lrelu(self.conv1(x))
#         y = self.lrelu(self.conv2(y))
#         y = self.downsample(y)
#         return y

# class Discriminator(nn.Module):
#     # CHANGED: Default num_channels is now 2
#     def __init__(self, num_channels=2, fmap_base=4096, fmap_max=256, max_res_log2=6):
#         super().__init__()
#         def nf(stage):
#             return min(int(fmap_base / (2.0 ** stage)), fmap_max)
#         self.prog_blocks = nn.ModuleList()
#         self.from_rgb_layers = nn.ModuleList()
#         for i in range(max_res_log2 - 1):
#             in_ch = nf(i)
#             out_ch = nf(i-1) if i > 0 else nf(0)
#             self.from_rgb_layers.append(nn.Sequential(WSConv3d(num_channels, in_ch, 1, 1, 0), nn.LeakyReLU(0.2)))
#             self.prog_blocks.append(D_Block(in_ch, out_ch))
#         self.final_block = nn.Sequential(
#             MinibatchStdDev(),
#             WSConv3d(nf(0) + 1, nf(0), 3, 1, 1), nn.LeakyReLU(0.2),
#             nn.Flatten(),
#             nn.Linear(nf(0) * (4**3), nf(0)), nn.LeakyReLU(0.2),
#             nn.Linear(nf(0), 1)
#         )
#     def forward(self, x, level, alpha):
#         level = int(level)
#         y = self.from_rgb_layers[level](x)
#         if level > 0 and alpha < 1.0:
#             y_prev = F.avg_pool3d(x, 2)
#             y_prev = self.from_rgb_layers[level - 1](y_prev)
#             y = self.prog_blocks[level-1](y)
#             y = (1.0 - alpha) * y_prev + alpha * y
#             for i in range(level - 1, 0, -1):
#                 y = self.prog_blocks[i-1](y)
#         else:
#             for i in range(level, 0, -1):
#                 y = self.prog_blocks[i-1](y)
#         score = self.final_block(y)
#         return score, None



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# # --- Helper Layers (Unchanged) ---
# class WSConv3d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, gain=2):
#         super().__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
#         self.scale = (gain / (in_channels * (kernel_size ** 3))) ** 0.5
#         self.bias = nn.Parameter(torch.zeros(out_channels))
#         nn.init.normal_(self.conv.weight)
#         nn.init.zeros_(self.bias)
#     def forward(self, x):
#         return self.conv(x * self.scale) + self.bias.view(1, -1, 1, 1, 1)

# class PixelNorm(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.epsilon = 1e-8
#     def forward(self, x):
#         return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

# class MinibatchStdDev(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x):
#         batch_size, _, depth, height, width = x.shape
#         y = x - torch.mean(x, dim=0, keepdim=True)
#         y = torch.mean(y ** 2, dim=0)
#         y = torch.sqrt(y + 1e-8)
#         y = torch.mean(y).view(1, 1, 1, 1, 1)
#         y = y.repeat(batch_size, 1, depth, height, width)
#         return torch.cat([x, y], dim=1)

# # --- Generator (Unchanged) ---
# class G_Block(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = WSConv3d(in_channels, out_channels, 3, 1, 1)
#         self.conv2 = WSConv3d(out_channels, out_channels, 3, 1, 1)
#         self.pixel_norm = PixelNorm()
#         self.lrelu = nn.LeakyReLU(0.2)
#     def forward(self, x):
#         x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
#         x = self.lrelu(self.pixel_norm(self.conv1(x)))
#         x = self.lrelu(self.pixel_norm(self.conv2(x)))
#         return x

# class Generator(nn.Module):
#     def __init__(self, latent_size=256, num_channels=1, fmap_base=4096, fmap_max=256, max_res_log2=6):
#         super().__init__()
#         def nf(stage):
#             return min(int(fmap_base / (2.0 ** stage)), fmap_max)
#         self.initial_block = nn.Sequential(
#             PixelNorm(),
#             nn.ConvTranspose3d(latent_size, nf(0), 4, 1, 0),
#             nn.LeakyReLU(0.2), PixelNorm(),
#             WSConv3d(nf(0), nf(0), 3, 1, 1),
#             nn.LeakyReLU(0.2), PixelNorm(),
#         )
#         self.prog_blocks = nn.ModuleList()
#         self.to_rgb_layers = nn.ModuleList()
#         self.to_rgb_layers.append(WSConv3d(nf(0), num_channels, 1, 1, 0))
#         for i in range(1, max_res_log2 - 1):
#             in_ch, out_ch = nf(i-1), nf(i)
#             self.prog_blocks.append(G_Block(in_ch, out_ch))
#             self.to_rgb_layers.append(WSConv3d(out_ch, num_channels, 1, 1, 0))
#     def forward(self, x, level, alpha):
#         level = int(level)
#         x = self.initial_block(x.view(x.shape[0], -1, 1, 1, 1))
#         if level == 0:
#             return self.to_rgb_layers[0](x)
#         for i in range(level):
#             x_prev = x
#             x = self.prog_blocks[i](x)
#         out_stable = self.to_rgb_layers[level](x)
#         if alpha < 1.0:
#             out_faded = F.interpolate(self.to_rgb_layers[level-1](x_prev), scale_factor=2)
#             return (1 - alpha) * out_faded + alpha * out_stable
#         return out_stable

# # --- Discriminator (DEFINITIVELY CORRECTED) ---
# class D_Block(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = WSConv3d(in_channels, in_channels, 3, 1, 1)
#         self.conv2 = WSConv3d(in_channels, out_channels, 3, 1, 1)
#         self.downsample = nn.AvgPool3d(2)
#         self.lrelu = nn.LeakyReLU(0.2)
#     def forward(self, x):
#         y = self.lrelu(self.conv1(x))
#         y = self.lrelu(self.conv2(y))
#         y = self.downsample(y)
#         return y

# class Discriminator(nn.Module):
#     def __init__(self, num_channels=1, fmap_base=4096, fmap_max=256, max_res_log2=6):
#         super().__init__()
#         def nf(stage):
#             return min(int(fmap_base / (2.0 ** stage)), fmap_max)

#         # Build layers from highest resolution down to lowest
#         self.prog_blocks = nn.ModuleList()
#         self.from_rgb_layers = nn.ModuleList()

#         for i in range(max_res_log2 - 2, -1, -1):
#             in_ch = nf(i + 1)
#             out_ch = nf(i)
#             self.prog_blocks.append(D_Block(in_ch, out_ch))
#             self.from_rgb_layers.append(nn.Sequential(WSConv3d(num_channels, in_ch, 1, 1, 0), nn.LeakyReLU(0.2)))
        
#         # Add the final from_rgb layer for the 4x4 resolution
#         self.from_rgb_layers.append(nn.Sequential(WSConv3d(num_channels, nf(0), 1, 1, 0), nn.LeakyReLU(0.2)))
        
#         # Reverse the from_rgb_layers list so it can be indexed simply by level (0 to N)
#         temp_list = list(self.from_rgb_layers)
#         temp_list.reverse()
#         self.from_rgb_layers = nn.ModuleList(temp_list)

#         self.final_block = nn.Sequential(
#             MinibatchStdDev(),
#             WSConv3d(nf(0) + 1, nf(0), 3, 1, 1), nn.LeakyReLU(0.2),
#             nn.Flatten(),
#             nn.Linear(nf(0) * (4**3), nf(0)), nn.LeakyReLU(0.2),
#             nn.Linear(nf(0), 1)
#         )
#         self.max_res_log2 = max_res_log2
        
#     def forward(self, x, level, alpha):
#         level = int(level)
        
#         # The main path starts with the high-resolution feature map
#         y = self.from_rgb_layers[level](x)

#         if level > 0 and alpha < 1.0: # Fade-in logic
#             # The main path is downsampled once
#             y_main_path = self.prog_blocks[len(self.prog_blocks) - level](y)
            
#             # The alternate path comes from the lower-resolution image
#             y_prev_path = F.avg_pool3d(x, 2)
#             y_prev_path = self.from_rgb_layers[level - 1](y_prev_path)
            
#             # Blend the two paths (they are now the same size)
#             y = (1.0 - alpha) * y_prev_path + alpha * y_main_path
            
#             # Apply the rest of the downsampling blocks
#             for i in range(len(self.prog_blocks) - level + 1, len(self.prog_blocks)):
#                 y = self.prog_blocks[i](y)
#         else: # Stabilization logic
#             # Apply all downsampling blocks sequentially
#             start_block_idx = len(self.prog_blocks) - level
#             for i in range(start_block_idx, len(self.prog_blocks)):
#                 y = self.prog_blocks[i](y)
            
#         score = self.final_block(y)
#         return score, None