#Two G-D
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import os
import math
import argparse
from torch.cuda.amp import autocast, GradScaler

from config import config, DATA_ROOT, RESULT_DIR
from dataset import DualChannelDataset
from networks import Generator_Baseline, Generator_Followup, Discriminator

def get_dataloader(resolution, batch_size, class_id=0, num_workers=16):
    dataset = DualChannelDataset(root_dir=DATA_ROOT, target_resolution=resolution, class_id=class_id)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

def compute_gradient_penalty(discriminator, real_samples, fake_samples, level, alpha, device):
    eps = torch.randn(real_samples.shape[0], 1, 1, 1, 1, device=device)
    interpolates = (eps * real_samples + (1 - eps) * fake_samples).requires_grad_(True)
    d_interpolates, _ = discriminator(interpolates, level, alpha)
    gradients = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates, device=device),
        create_graph=True, retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty

def compute_ssim_loss(img1, img2, window_size=11, size_average=True):
    """
    Compute SSIM loss between two images.
    SSIM measures structural similarity - we want followup to maintain structure from baseline.
    """
    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    gauss = gauss / gauss.sum()
    
    # Create 3D window properly
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    
    # Create 3D window by outer product: [ws, ws, 1] * [1, ws]
    _3D_window = _2D_window.unsqueeze(2) * gauss.unsqueeze(0).unsqueeze(0)
    
    window = _3D_window.unsqueeze(0).unsqueeze(0)  # [1, 1, ws, ws, ws]
    window = window.expand(img1.size(1), 1, window_size, window_size, window_size).contiguous()
    window = window.to(img1.device).type_as(img1)
    
    mu1 = F.conv3d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv3d(img2, window, padding=window_size//2, groups=img2.size(1))
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return 1 - ssim_map.mean()  # Return as loss (lower is better)
    else:
        return 1 - ssim_map.mean(1).mean(1).mean(1)

def save_3plane_samples(samples, save_path, nrow=4, normalize=True):
    """
    Save axial, sagittal, and coronal slices of 3D volumes.
    
    Args:
        samples: [batch, channels, D, H, W] tensor
        save_path: base path (e.g., "sample_100k_baseline.png")
        nrow: number of images per row
        normalize: whether to normalize for visualization
    """
    batch, channels, D, H, W = samples.shape
    
    # Get middle slices for each plane
    mid_d = D // 2  # Axial
    mid_h = H // 2  # Coronal
    mid_w = W // 2  # Sagittal
    
    # Extract slices
    axial_slice = samples[:, :, mid_d, :, :]      # [batch, 1, H, W]
    sagittal_slice = samples[:, :, :, :, mid_w]   # [batch, 1, D, H]
    coronal_slice = samples[:, :, :, mid_h, :]    # [batch, 1, D, W]
    
    # Create file paths
    base_path = save_path.replace('.png', '')
    axial_path = f"{base_path}_axial.png"
    sagittal_path = f"{base_path}_sagittal.png"
    coronal_path = f"{base_path}_coronal.png"
    
    # Save each plane
    torchvision.utils.save_image(axial_slice, axial_path, nrow=nrow, normalize=normalize)
    torchvision.utils.save_image(sagittal_slice, sagittal_path, nrow=nrow, normalize=normalize)
    torchvision.utils.save_image(coronal_slice, coronal_path, nrow=nrow, normalize=normalize)
    
    print(f"    Saved 3-plane samples: {base_path}_[axial/sagittal/coronal].png")

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    use_amp = args.use_amp and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    print(f"Mixed precision training: {'Enabled' if use_amp else 'Disabled'}")
    
    class_to_train = args.class_id
    print(f"--- Starting Two-Generator Training for Class {class_to_train} ---")
    print(f"--- Generator_Baseline: noise → baseline ---")
    print(f"--- Generator_Followup: baseline + noise → follow-up (with L1 + SSIM consistency) ---")
    print(f"--- Resolution: {config.IMAGE_SIZE}³ ---")
    
    current_kimg, stage, level = 0, 0, 0
    max_res, max_res_log2 = config.IMAGE_SIZE, int(math.log2(config.IMAGE_SIZE))
    
    # Initialize TWO generators
    g_baseline = Generator_Baseline(config.LATENT_SIZE, config.NUM_CHANNELS, 
                                     config.FMAP_BASE, config.FMAP_MAX, max_res_log2).to(device)
    g_followup = Generator_Followup(config.LATENT_SIZE, config.NUM_CHANNELS,
                                    config.FMAP_BASE, config.FMAP_MAX, max_res_log2).to(device)
    
    # Initialize TWO discriminators
    d_baseline = Discriminator(config.NUM_CHANNELS, config.FMAP_BASE, config.FMAP_MAX, max_res_log2).to(device)
    d_followup = Discriminator(config.NUM_CHANNELS, config.FMAP_BASE, config.FMAP_MAX, max_res_log2).to(device)
    
    # EMA versions
    g_baseline_ema = Generator_Baseline(config.LATENT_SIZE, config.NUM_CHANNELS,
                                        config.FMAP_BASE, config.FMAP_MAX, max_res_log2).to(device)
    g_followup_ema = Generator_Followup(config.LATENT_SIZE, config.NUM_CHANNELS,
                                        config.FMAP_BASE, config.FMAP_MAX, max_res_log2).to(device)
    g_baseline_ema.eval()
    g_followup_ema.eval()
    
    # Optimizers
    g_baseline_optimizer = optim.Adam(g_baseline.parameters(), lr=config.LR_BASE, betas=config.ADAM_BETAS)
    g_followup_optimizer = optim.Adam(g_followup.parameters(), lr=config.LR_BASE, betas=config.ADAM_BETAS)
    d_baseline_optimizer = optim.Adam(d_baseline.parameters(), lr=config.LR_BASE, betas=config.ADAM_BETAS)
    d_followup_optimizer = optim.Adam(d_followup.parameters(), lr=config.LR_BASE, betas=config.ADAM_BETAS)

    if args.resume_from:
        print(f"--- Resuming training from checkpoint: {args.resume_from} ---")
        checkpoint = torch.load(args.resume_from, map_location=device)
        g_baseline.load_state_dict(checkpoint['g_baseline'])
        g_followup.load_state_dict(checkpoint['g_followup'])
        d_baseline.load_state_dict(checkpoint['d_baseline'])
        d_followup.load_state_dict(checkpoint['d_followup'])
        g_baseline_ema.load_state_dict(checkpoint['g_baseline_ema'])
        g_followup_ema.load_state_dict(checkpoint['g_followup_ema'])
        g_baseline_optimizer.load_state_dict(checkpoint['g_baseline_optimizer'])
        g_followup_optimizer.load_state_dict(checkpoint['g_followup_optimizer'])
        d_baseline_optimizer.load_state_dict(checkpoint['d_baseline_optimizer'])
        d_followup_optimizer.load_state_dict(checkpoint['d_followup_optimizer'])
        level = checkpoint['level']
        stage = checkpoint['stage']
        current_kimg = checkpoint['current_kimg']
        print(f"--- Resumed at k-img: {current_kimg:.2f}, Level: {level}, Stage: {stage} ---")

    output_dir = os.path.join(RESULT_DIR, f"class_{class_to_train}_two_generators_256")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    
    # Fixed noise for visualization
    fixed_noise_baseline = torch.randn(8, config.LATENT_SIZE, device=device)
    fixed_noise_followup = torch.randn(8, config.LATENT_SIZE, device=device)
    
    # Loss weights for G_followup
    lambda_l1 = args.lambda_l1        # Weight for L1 loss (pixel-wise consistency)
    lambda_ssim = args.lambda_ssim    # Weight for SSIM loss (structural consistency)
    
    print(f"--- Loss weights: λ_L1={lambda_l1}, λ_SSIM={lambda_ssim} ---")
    
    next_save_kimg = (int(current_kimg / config.SAVE_INTERVAL_KIMG) + 1) * config.SAVE_INTERVAL_KIMG

    while current_kimg < config.TOTAL_KIMG:
        resolution = 4 * (2 ** level)
        batch_size = config.BATCH_SIZES.get(resolution, 1)  # Handle 256 resolution
        current_stage_name = "Fade-in" if stage == 1 else "Stabilize"
        print(f"\nK-img: {current_kimg:.2f} | Level: {level} ({resolution}³) | Stage: {current_stage_name} | Batch Size: {batch_size}")
        
        num_workers = min(args.num_workers, 12) if resolution <= 32 else min(args.num_workers, 8)
        dataloader = get_dataloader(resolution, batch_size, class_id=class_to_train, num_workers=num_workers)
        
        kimg_processed_in_stage = 0
        for epoch in range(500):
            if kimg_processed_in_stage >= config.KIMG_PER_STAGE:
                break

            for step, (real_baseline, real_followup, _) in enumerate(dataloader):
                alpha = kimg_processed_in_stage / config.KIMG_PER_STAGE if stage == 1 else 1.0
                
                g_baseline.train()
                g_followup.train()
                d_baseline.train()
                d_followup.train()
                
                real_baseline = real_baseline.to(device, non_blocking=True)
                real_followup = real_followup.to(device, non_blocking=True)
                
                # --- Train Discriminator_Baseline ---
                d_baseline_optimizer.zero_grad()
                noise_baseline = torch.randn(batch_size, config.LATENT_SIZE, device=device)
                
                if use_amp:
                    with autocast():
                        fake_baseline = g_baseline(noise_baseline, level, alpha).detach()
                        d_real_baseline, _ = d_baseline(real_baseline, level, alpha)
                        d_fake_baseline, _ = d_baseline(fake_baseline, level, alpha)
                        gp_baseline = compute_gradient_penalty(d_baseline, real_baseline, fake_baseline, level, alpha, device)
                        d_baseline_loss = d_fake_baseline.mean() - d_real_baseline.mean() + config.LAMBDA_GP * gp_baseline
                    scaler.scale(d_baseline_loss).backward()
                    scaler.step(d_baseline_optimizer)
                    scaler.update()
                else:
                    fake_baseline = g_baseline(noise_baseline, level, alpha).detach()
                    d_real_baseline, _ = d_baseline(real_baseline, level, alpha)
                    d_fake_baseline, _ = d_baseline(fake_baseline, level, alpha)
                    gp_baseline = compute_gradient_penalty(d_baseline, real_baseline, fake_baseline, level, alpha, device)
                    d_baseline_loss = d_fake_baseline.mean() - d_real_baseline.mean() + config.LAMBDA_GP * gp_baseline
                    d_baseline_loss.backward()
                    d_baseline_optimizer.step()

                # --- Train Discriminator_Followup ---
                d_followup_optimizer.zero_grad()
                noise_baseline_for_followup = torch.randn(batch_size, config.LATENT_SIZE, device=device)
                noise_followup = torch.randn(batch_size, config.LATENT_SIZE, device=device)
                
                if use_amp:
                    with autocast():
                        # Generate synthetic baseline, then synthetic follow-up
                        synthetic_baseline = g_baseline(noise_baseline_for_followup, level, alpha).detach()
                        fake_followup = g_followup(noise_followup, synthetic_baseline, level, alpha).detach()
                        d_real_followup, _ = d_followup(real_followup, level, alpha)
                        d_fake_followup, _ = d_followup(fake_followup, level, alpha)
                        gp_followup = compute_gradient_penalty(d_followup, real_followup, fake_followup, level, alpha, device)
                        d_followup_loss = d_fake_followup.mean() - d_real_followup.mean() + config.LAMBDA_GP * gp_followup
                    scaler.scale(d_followup_loss).backward()
                    scaler.step(d_followup_optimizer)
                    scaler.update()
                else:
                    synthetic_baseline = g_baseline(noise_baseline_for_followup, level, alpha).detach()
                    fake_followup = g_followup(noise_followup, synthetic_baseline, level, alpha).detach()
                    d_real_followup, _ = d_followup(real_followup, level, alpha)
                    d_fake_followup, _ = d_followup(fake_followup, level, alpha)
                    gp_followup = compute_gradient_penalty(d_followup, real_followup, fake_followup, level, alpha, device)
                    d_followup_loss = d_fake_followup.mean() - d_real_followup.mean() + config.LAMBDA_GP * gp_followup
                    d_followup_loss.backward()
                    d_followup_optimizer.step()

                # --- Train Generator_Baseline ---
                g_baseline_optimizer.zero_grad()
                noise_baseline = torch.randn(batch_size, config.LATENT_SIZE, device=device)
                
                if use_amp:
                    with autocast():
                        fake_baseline = g_baseline(noise_baseline, level, alpha)
                        g_fake_baseline, _ = d_baseline(fake_baseline, level, alpha)
                        g_baseline_loss = -g_fake_baseline.mean()
                    scaler.scale(g_baseline_loss).backward()
                    scaler.step(g_baseline_optimizer)
                    scaler.update()
                else:
                    fake_baseline = g_baseline(noise_baseline, level, alpha)
                    g_fake_baseline, _ = d_baseline(fake_baseline, level, alpha)
                    g_baseline_loss = -g_fake_baseline.mean()
                    g_baseline_loss.backward()
                    g_baseline_optimizer.step()

                # --- Train Generator_Followup (WITH L1 + SSIM CONSISTENCY LOSS) ---
                g_followup_optimizer.zero_grad()
                noise_baseline_for_followup = torch.randn(batch_size, config.LATENT_SIZE, device=device)
                noise_followup = torch.randn(batch_size, config.LATENT_SIZE, device=device)
                
                if use_amp:
                    with autocast():
                        synthetic_baseline = g_baseline(noise_baseline_for_followup, level, alpha).detach()
                        fake_followup = g_followup(noise_followup, synthetic_baseline, level, alpha)
                        g_fake_followup, _ = d_followup(fake_followup, level, alpha)
                        
                        # Adversarial loss
                        adversarial_loss = -g_fake_followup.mean()
                        
                        # L1 loss: Encourage structural similarity with baseline
                        # (followup should be similar but not identical)
                        l1_loss = F.l1_loss(fake_followup, synthetic_baseline)
                        
                        # SSIM loss: Preserve structural information from baseline
                        ssim_loss = compute_ssim_loss(fake_followup, synthetic_baseline)
                        
                        # Combined loss
                        g_followup_loss = adversarial_loss + lambda_l1 * l1_loss + lambda_ssim * ssim_loss
                    
                    scaler.scale(g_followup_loss).backward()
                    scaler.step(g_followup_optimizer)
                    scaler.update()
                else:
                    synthetic_baseline = g_baseline(noise_baseline_for_followup, level, alpha).detach()
                    fake_followup = g_followup(noise_followup, synthetic_baseline, level, alpha)
                    g_fake_followup, _ = d_followup(fake_followup, level, alpha)
                    
                    # Adversarial loss
                    adversarial_loss = -g_fake_followup.mean()
                    
                    # L1 loss
                    l1_loss = F.l1_loss(fake_followup, synthetic_baseline)
                    
                    # SSIM loss
                    ssim_loss = compute_ssim_loss(fake_followup, synthetic_baseline)
                    
                    # Combined loss
                    g_followup_loss = adversarial_loss + lambda_l1 * l1_loss + lambda_ssim * ssim_loss
                    
                    g_followup_loss.backward()
                    g_followup_optimizer.step()

                # EMA updates
                with torch.no_grad():
                    for p_ema, p in zip(g_baseline_ema.parameters(), g_baseline.parameters()):
                        p_ema.data.mul_(0.999).add_((1 - 0.999) * p.data)
                    for p_ema, p in zip(g_followup_ema.parameters(), g_followup.parameters()):
                        p_ema.data.mul_(0.999).add_((1 - 0.999) * p.data)

                current_kimg += batch_size / 1000.0
                kimg_processed_in_stage += batch_size / 1000.0
                
                if step % config.LOG_INTERVAL_STEPS == 0:
                    print(f"  Epoch: {epoch} | Batch: {step}/{len(dataloader)}")
                    print(f"    D_Baseline: {d_baseline_loss.item():.3f} | G_Baseline: {g_baseline_loss.item():.3f}")
                    if not use_amp:
                        print(f"    D_Followup: {d_followup_loss.item():.3f} | G_Followup: {g_followup_loss.item():.3f}")
                        print(f"      └─ Adv: {adversarial_loss.item():.3f}, L1: {l1_loss.item():.3f}, SSIM: {ssim_loss.item():.3f} | Alpha: {alpha:.2f}")
                    else:
                        print(f"    D_Followup: {d_followup_loss.item():.3f} | G_Followup: {g_followup_loss.item():.3f} | Alpha: {alpha:.2f}")

                # Save checkpoints and samples
                if current_kimg >= next_save_kimg:
                    g_baseline_ema.eval()
                    g_followup_ema.eval()
                    with torch.no_grad():
                        # Generate paired samples using SAME noise
                        fake_baselines = g_baseline_ema(fixed_noise_baseline, level, alpha)
                        fake_followups = g_followup_ema(fixed_noise_followup, fake_baselines, level, alpha)
                        
                        print(f"\n  Saving 3-plane visualizations...")
                        
                        # Save baseline samples (all 3 planes)
                        baseline_path = os.path.join(output_dir, "samples", f"sample_{int(current_kimg)}k_res{resolution}_baseline.png")
                        save_3plane_samples(fake_baselines, baseline_path, nrow=4, normalize=True)
                        
                        # Save follow-up samples (all 3 planes)
                        followup_path = os.path.join(output_dir, "samples", f"sample_{int(current_kimg)}k_res{resolution}_followup.png")
                        save_3plane_samples(fake_followups, followup_path, nrow=4, normalize=True)
                        
                        # Save difference map (all 3 planes)
                        differences = fake_followups - fake_baselines
                        diff_path = os.path.join(output_dir, "samples", f"sample_{int(current_kimg)}k_res{resolution}_difference.png")
                        save_3plane_samples(differences, diff_path, nrow=4, normalize=True)
                    
                    torch.save({
                        'g_baseline': g_baseline.state_dict(),
                        'g_followup': g_followup.state_dict(),
                        'd_baseline': d_baseline.state_dict(),
                        'd_followup': d_followup.state_dict(),
                        'g_baseline_ema': g_baseline_ema.state_dict(),
                        'g_followup_ema': g_followup_ema.state_dict(),
                        'g_baseline_optimizer': g_baseline_optimizer.state_dict(),
                        'g_followup_optimizer': g_followup_optimizer.state_dict(),
                        'd_baseline_optimizer': d_baseline_optimizer.state_dict(),
                        'd_followup_optimizer': d_followup_optimizer.state_dict(),
                        'level': level,
                        'stage': stage,
                        'current_kimg': current_kimg,
                        'lambda_l1': lambda_l1,
                        'lambda_ssim': lambda_ssim,
                    }, os.path.join(output_dir, f"checkpoint_{int(current_kimg)}k.pth"))
                    print(f"--- Checkpoint saved at {current_kimg:.2f} k-images ---\n")
                    next_save_kimg += config.SAVE_INTERVAL_KIMG
                
                if kimg_processed_in_stage >= config.KIMG_PER_STAGE:
                    break
        
        # Stage Progression Logic
        if level == 0 and stage == 0:
            stage = 1
            level += 1
        elif stage == 0:
            stage = 1
            level += 1
        elif stage == 1:
            stage = 0
        if level >= max_res_log2 - 1:
            level = max_res_log2 - 2
            stage = 0

    print("--- Training Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train two-generator PGGAN for paired baseline/follow-up generation.")
    parser.add_argument("--class_id", type=int, required=True, help="The class ID to train the GAN on.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint .pth file.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers (default: 8).")
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision training for speedup.")
    parser.add_argument("--lambda_l1", type=float, default=10.0, help="Weight for L1 consistency loss (default: 10.0)")
    parser.add_argument("--lambda_ssim", type=float, default=1.0, help="Weight for SSIM consistency loss (default: 1.0)")
    args = parser.parse_args()
    train(args)
# # train.py
# import torch
# import torch.optim as optim
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import torchvision
# import os
# import argparse
# from tqdm import tqdm
# import math

# from config import config, DATA_ROOT, RESULT_DIR
# from dataset import LumiereDataset
# from networks import Generator, Discriminator

# def get_dataloader(resolution, batch_size, class_id=0):
#     """Initializes and returns a DataLoader for a specific class and resolution."""
#     dataset = LumiereDataset(root_dir=DATA_ROOT, target_resolution=resolution, class_id=class_id)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

# def compute_gradient_penalty(discriminator, real_samples, fake_samples, level, alpha, device):
#     """Calculates the gradient penalty loss for WGAN GP."""
#     eps = torch.randn(real_samples.shape[0], 1, 1, 1, 1, device=device)
#     interpolates = (eps * real_samples + (1 - eps) * fake_samples).requires_grad_(True)
#     d_interpolates, _ = discriminator(interpolates, level, alpha)
#     gradients = torch.autograd.grad(
#         outputs=d_interpolates, inputs=interpolates,
#         grad_outputs=torch.ones_like(d_interpolates, device=device),
#         create_graph=True, retain_graph=True,
#     )[0]
#     gradients = gradients.view(gradients.size(0), -1)
#     gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
#     return gradient_penalty

# def train(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # --- 1. Prepare Models and Optimizers ---
#     max_res_log2 = int(math.log2(config.IMAGE_SIZE))
#     generator = Generator(config.LATENT_SIZE, config.NUM_CHANNELS, config.FMAP_BASE, config.FMAP_MAX, max_res_log2).to(device)
#     discriminator = Discriminator(config.NUM_CHANNELS, config.FMAP_BASE, config.FMAP_MAX, max_res_log2).to(device)
#     g_ema = Generator(config.LATENT_SIZE, config.NUM_CHANNELS, config.FMAP_BASE, config.FMAP_MAX, max_res_log2).to(device)
#     g_ema.eval()

#     optimizer_G = optim.Adam(generator.parameters(), lr=config.LR_BASE, betas=config.ADAM_BETAS)
#     optimizer_D = optim.Adam(discriminator.parameters(), lr=config.LR_BASE, betas=config.ADAM_BETAS)
    
#     current_kimg = 0
#     if args.resume_from:
#         print(f"--- Resuming training from: {args.resume_from} ---")
#         checkpoint = torch.load(args.resume_from, map_location=device)
#         generator.load_state_dict(checkpoint['generator'], strict=False)
#         discriminator.load_state_dict(checkpoint['discriminator'], strict=False)
#         g_ema.load_state_dict(checkpoint['g_ema'], strict=False)
#         optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
#         optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
#         current_kimg = checkpoint.get('current_kimg', 0.0)
#         print(f"--- Resumed at k-img: {current_kimg:.2f} ---")

#     output_dir = os.path.join(RESULT_DIR, f"class_{args.class_id}")
#     os.makedirs(output_dir, exist_ok=True); os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
#     fixed_noise = torch.randn(8, config.LATENT_SIZE, device=device)
    
#     next_save_kimg = (int(current_kimg / config.SAVE_INTERVAL_KIMG) + 1) * config.SAVE_INTERVAL_KIMG

#     # --- 3. Main Training Loop ---
#     print("\n--- Starting PGGAN Training ---")
#     while current_kimg < config.TOTAL_KIMG:
        
#         # --- Robustly calculate the current level, stage, and alpha ---
#         kimg_per_phase = config.KIMG_PER_STAGE
#         completed_phases = int(current_kimg / kimg_per_phase)
        
#         level = completed_phases // 2
#         stage = completed_phases % 2 # 0 for Stabilize, 1 for Fade-in
        
#         # The very first stage is always stabilization at level 0
#         if level == 0:
#             stage = 0
        
#         # Cap level at the max resolution
#         level = min(level, max_res_log2 - 2)
        
#         alpha = 1.0
#         current_stage_name = "Stabilize"
#         if stage == 1 and level > 0:
#             kimg_into_phase = current_kimg % kimg_per_phase
#             alpha = kimg_into_phase / kimg_per_phase
#             current_stage_name = "Fade-in"

#         resolution = 4 * (2 ** level)
#         batch_size = config.BATCH_SIZES[resolution]
        
#         print(f"\nK-img: {current_kimg:.2f} | Level: {level} ({resolution}x{resolution}) | Stage: {current_stage_name} | Batch Size: {batch_size}")
        
#         dataloader = get_dataloader(resolution, batch_size, class_id=args.class_id)
        
#         kimg_at_stage_start = current_kimg
        
#         # --- NEW: Corrected Epoch Loop ---
#         for epoch in range(1000): # High number, will be broken out of
#             if current_kimg - kimg_at_stage_start >= kimg_per_phase:
#                 break
                
#             loop = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=True)
#             for step, (real_images, _) in enumerate(loop):
#                 if current_kimg - kimg_at_stage_start >= kimg_per_phase:
#                     break
                
#                 real_images = real_images.to(device)
                
#                 # --- Train Discriminator ---
#                 optimizer_D.zero_grad()
#                 noise = torch.randn(batch_size, config.LATENT_SIZE, device=device)
#                 fake_images = generator(noise, level, alpha).detach()
#                 d_real, _ = discriminator(real_images, level, alpha)
#                 d_fake, _ = discriminator(fake_images, level, alpha)
#                 gp = compute_gradient_penalty(discriminator, real_images, fake_images, level, alpha, device)
#                 d_loss = d_fake.mean() - d_real.mean() + config.LAMBDA_GP * gp
#                 d_loss.backward()
#                 optimizer_D.step()
                
#                 # --- Train Generator ---
#                 optimizer_G.zero_grad()
#                 noise = torch.randn(batch_size, config.LATENT_SIZE, device=device)
#                 fake_images = generator(noise, level, alpha)
#                 g_fake, _ = discriminator(fake_images, level, alpha)
#                 g_loss = -g_fake.mean()
#                 g_loss.backward()
#                 optimizer_G.step()
                
#                 with torch.no_grad():
#                     for p_ema, p in zip(g_ema.parameters(), generator.parameters()):
#                         p_ema.data.mul_(0.999).add_((1 - 0.999) * p.data)

#                 current_kimg += batch_size / 1000.0
                
#                 loop.set_postfix(D_loss=d_loss.item(), G_loss=g_loss.item(), alpha=alpha)

#                 # --- Saving Logic ---
#                 if current_kimg >= next_save_kimg:
#                     # (Saving logic is unchanged from previous correct version)
#                     pass

#     print("--- Training Finished ---")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train a PGGAN on longitudinal MRI data.")
#     parser.add_argument("--class_id", type=int, required=True, help="The class ID to train the GAN on.")
#     parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint .pth file to resume from.")
#     args = parser.parse_args()
#     train(args)