# Two G-D

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with attribute syntax."""
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
    def __getattr__(self, name): 
        return self[name]
    def __setattr__(self, name, value): 
        self[name] = value
    def __delattr__(self, name): 
        del self[name]

# --- Project Paths ---
DATA_ROOT = 'CACHED256'
RESULT_DIR = './saved256'

# --- Training Hyperparameters (OPTIMIZED FOR SPEED) ---
config = EasyDict(
    # Dataset and DataLoader
    IMAGE_SIZE = 256,  # UPDATED to 256
    NUM_CHANNELS = 1,  # Single channel (either baseline OR follow-up)
    NUM_CLASSES = 4,   # Your original response classes
    BATCH_SIZES = {
        4: 16, 
        8: 16, 
        16: 8, 
        32: 4, 
        64: 2, 
        128: 1,
        256: 1   # Added for 256 resolution
    },  # Original batch sizes + 256

    # BATCH_SIZES = { 4: 4 , 8: 4 , 16: 4, 32: 4, 64: 2, 128: 1, 256: 1},
    
    # Model and Optimizer (KEEPING ORIGINAL)
    LATENT_SIZE = 256,
    FMAP_BASE = 1024,  # Original value
    FMAP_MAX = 256,    # Original value
    LR_BASE = 0.0015,  # Original value
    ADAM_BETAS = (0.0, 0.99),
    
    # --- Training Schedule (ONLY THESE 3 REDUCED FOR SPEED) ---
    TOTAL_KIMG = 1500,         # REDUCED from 2000
    KIMG_PER_STAGE = 75,       # REDUCED from 100
    
    # WGAN-GP Loss
    LAMBDA_GP = 10,
    
    # Logging and Saving
    SAVE_INTERVAL_KIMG = 10,   # REDUCED from 100
    LOG_INTERVAL_STEPS = 50,
)
# class EasyDict(dict):
#     """Convenience class that behaves like a dict but allows access with attribute syntax."""
#     def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
#     def __getattr__(self, name): return self[name]
#     def __setattr__(self, name, value): self[name] = value
#     def __delattr__(self, name): del self[name]

# # --- Project Paths ---
# DATA_ROOT = 'CACHED'
# RESULT_DIR = './saved'

# # --- Training Hyperparameters ---
# config = EasyDict(
#     IMAGE_SIZE = 64,
#     # This is a 2-channel model: (Baseline T1ce, Difference Map)
#     NUM_CHANNELS = 2,
#     NUM_CLASSES = 4,
#     BATCH_SIZES = {4: 16, 8: 16, 16: 8, 32: 4, 64: 2},
    
#     LATENT_SIZE = 256,
#     FMAP_BASE = 4096,
#     FMAP_MAX = 256,
#     LR_BASE = 0.0015,
#     ADAM_BETAS = (0.0, 0.99),
    
#     TOTAL_KIMG = 5000,
#     KIMG_PER_STAGE = 300,
    
#     LAMBDA_GP = 10,
    
#     SAVE_INTERVAL_KIMG = 50,
#     LOG_INTERVAL_STEPS = 50,
# )
# --- Performance Notes ---
# Training schedule optimizations for faster completion:
# 1. TOTAL_KIMG: 2000 -> 1500 (25% less training time)
# 2. KIMG_PER_STAGE: 100 -> 75 (25% faster progression between resolutions)
# 3. SAVE_INTERVAL_KIMG: 100 -> 50 (saves checkpoints more frequently)
#
# Architecture kept at original settings for full quality
# Estimated speedup: ~25-30% faster training overall
# class EasyDict(dict):
#     """Convenience class that behaves like a dict but allows access with attribute syntax."""
#     def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
#     def __getattr__(self, name): return self[name]
#     def __setattr__(self, name, value): self[name] = value
#     def __delattr__(self, name): del self[name]

# # --- Project Paths ---
# DATA_ROOT = 'CACHED'
# RESULT_DIR = './saved_debugged'

# # --- Training Hyperparameters ---
# config = EasyDict(
#     # Dataset and DataLoader
#     IMAGE_SIZE = 128, # Target the full resolution
#     NUM_CHANNELS = 1,
#     NUM_CLASSES = 4,
#     BATCH_SIZES = {4: 16, 8: 16, 16: 8, 32: 4, 64: 2, 128: 1},
    
#     # Model and Optimizer
#     LATENT_SIZE = 256,
#     FMAP_BASE = 1024, # Keep network thin for speed
#     FMAP_MAX = 256,
#     LR_BASE = 0.0015,
#     ADAM_BETAS = (0.0, 0.99),
    
#     # --- Training Schedule (MODIFIED FOR A VERY FAST DEBUG RUN) ---
    
#     # CHANGED: Just enough to complete all stages.
#     TOTAL_KIMG = 6,
    
#     # CHANGED: Make each stage last for only 500 images.
#     KIMG_PER_STAGE = 0.5,
    
#     # WGAN-GP Loss
#     LAMBDA_GP = 10,
    
#     # Logging and Saving
#     # CHANGED: Save a checkpoint at the end of every stage.
#     SAVE_INTERVAL_KIMG = 0.5,
    
#     LOG_INTERVAL_STEPS = 10, # Log more frequently
# )

# #H - code
# class EasyDict(dict):
#     """Convenience class that behaves like a dict but allows access with attribute syntax."""
#     def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
#     def __getattr__(self, name): return self[name]
#     def __setattr__(self, name, value): self[name] = value
#     def __delattr__(self, name): del self[name]

# # --- Project Paths ---
# # CHANGED: Point to the directory with your .npz files
# DATA_ROOT = 'CACHED'
# RESULT_DIR = './saved'

# # --- Training Hyperparameters ---
# config = EasyDict(
#     # Dataset and DataLoader
#     IMAGE_SIZE = 64,
#     # CHANGED: We are now training on a 2-channel tensor (BL T1ce + FU T1ce)
#     NUM_CHANNELS = 2,
#     NUM_CLASSES = 4,
#     BATCH_SIZES = {4: 16, 8: 16, 16: 8, 32: 4, 64: 2},
    
#     # Model and Optimizer
#     LATENT_SIZE = 256,
#     FMAP_BASE = 4096,
#     FMAP_MAX = 256,
#     LR_BASE = 0.0015,
#     ADAM_BETAS = (0.0, 0.99),
    
#     # Training Schedule for a good quality run
#     TOTAL_KIMG = 5000,
#     KIMG_PER_STAGE = 300,
    
#     # WGAN-GP Loss
#     LAMBDA_GP = 10,
    
#     # Logging and Saving
#     SAVE_INTERVAL_KIMG = 100,
#     LOG_INTERVAL_STEPS = 50,
# )
# class EasyDict(dict):
#     """Convenience class that behaves like a dict but allows access with attribute syntax."""
#     def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
#     def __getattr__(self, name): return self[name]
#     def __setattr__(self, name, value): self[name] = value
#     def __delattr__(self, name): del self[name]

# # --- Project Paths ---
# DATA_ROOT = 'CACHED'
# RESULT_DIR = './saved_debugged'

# # --- Training Hyperparameters ---
# config = EasyDict(
#     # Dataset and DataLoader
#     IMAGE_SIZE =128, # Keeping at 64 is good for speed
#     NUM_CHANNELS = 1,
#     NUM_CLASSES = 4,
#     BATCH_SIZES = {4: 16, 8: 16, 16: 8, 32: 4, 64: 2,128:1},
    
#     # Model and Optimizer
#     LATENT_SIZE = 256,
#     FMAP_BASE = 1024,
#     FMAP_MAX = 256,
#     LR_BASE = 0.0015,
#     ADAM_BETAS = (0.0, 0.99),
    
#     # --- Training Schedule (MODIFIED FOR SPEED) ---
#     # Total training duration, reduced from 2500
#     TOTAL_KIMG = 2000,
#     # Duration of each stage, reduced from 200. This is the lowest recommended value.
#     KIMG_PER_STAGE = 100,
    
#     # WGAN-GP Loss
#     LAMBDA_GP = 10,
    
#     # Logging and Saving
#     SAVE_INTERVAL_KIMG = 100, # Save more frequently
#     LOG_INTERVAL_STEPS = 50,
# )