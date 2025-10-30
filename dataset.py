#Two G-D
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class DualChannelDataset(Dataset):
    """
    Dataset that returns BOTH baseline and follow-up for each patient.
    Used to train both generators simultaneously.
    """
    def __init__(self, root_dir, target_resolution=64, class_id=0):
        self.root_dir = os.path.join(root_dir, str(class_id))
        self.resolution = target_resolution
        self.class_id = class_id
        
        print(f"[Dual Channel Dataset] Scanning for patient data in: {self.root_dir}...")
        self.file_paths = self._find_npz_files()
        if not self.file_paths:
            raise FileNotFoundError(f"No .npz files found for class {class_id} in {self.root_dir}")
        print(f"[Dual Channel Dataset] Found {len(self.file_paths)} patients for class {class_id}.")
        
        # Data multiplicity for speed
        self.real_size = len(self.file_paths)
        self.virtual_size = self.real_size * 100
    
    def _find_npz_files(self):
        paths = []
        for file_name in sorted(os.listdir(self.root_dir)):
            if file_name.startswith('.') or not file_name.endswith('.npz'):
                continue
            path = os.path.join(self.root_dir, file_name)
            paths.append(path)
        return paths
    
    def __len__(self):
        return self.virtual_size
    
    def __getitem__(self, idx):
        real_idx = idx % self.real_size
        file_path = self.file_paths[real_idx]
        
        try:
            data = np.load(file_path)
            tensor_10ch = data['vol']
            
            # Extract BOTH channels: baseline (1) and follow-up (5)
            baseline_channel = tensor_10ch[1].astype(np.float32)
            followup_channel = tensor_10ch[5].astype(np.float32)
            
            # Normalize each channel independently to [-1, 1]
            def normalize_channel(channel):
                min_val, max_val = np.min(channel), np.max(channel)
                if max_val - min_val > 0:
                    return 2 * (channel - min_val) / (max_val - min_val) - 1
                return channel
            
            baseline_normalized = normalize_channel(baseline_channel)
            followup_normalized = normalize_channel(followup_channel)
            
            # Convert to tensors [1, 1, D, H, W]
            baseline_tensor = torch.from_numpy(baseline_normalized).unsqueeze(0).unsqueeze(0)
            followup_tensor = torch.from_numpy(followup_normalized).unsqueeze(0).unsqueeze(0)
            
            # Resize both to target resolution
            baseline_resized = F.interpolate(
                baseline_tensor, 
                size=(self.resolution, self.resolution, self.resolution), 
                mode='trilinear', 
                align_corners=False
            ).squeeze(0)  # [1, res, res, res]
            
            followup_resized = F.interpolate(
                followup_tensor, 
                size=(self.resolution, self.resolution, self.resolution), 
                mode='trilinear', 
                align_corners=False
            ).squeeze(0)  # [1, res, res, res]
            
            # Return both baseline and follow-up
            return baseline_resized, followup_resized, self.class_id
        
        except Exception as e:
            print(f"Error loading data from file {file_path}: {e}")
            # Return zero tensors for both channels
            return (torch.zeros(1, self.resolution, self.resolution, self.resolution), 
                    torch.zeros(1, self.resolution, self.resolution, self.resolution),
                    self.class_id)
# #D-code
# # dataset.py
# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import torch.nn.functional as F
# import math

# class LumiereDataset(Dataset):
#     """
#     Dataset for loading a 2-channel tensor from pre-processed .npz files.
#     Channel 1: Baseline T1ce
#     Channel 2: (Follow-up T1ce) - (Baseline T1ce)
#     """
#     def __init__(self, root_dir, target_resolution=64, class_id=0):
#         self.root_dir = os.path.join(root_dir, str(class_id))
#         self.resolution = target_resolution
#         self.class_id = class_id
        
#         print(f"Scanning for patient data in: {self.root_dir}...")
#         self.file_paths = self._find_npz_files()
#         if not self.file_paths:
#             raise FileNotFoundError(f"No .npz files found for class {class_id} in {self.root_dir}")
#         print(f"Found {len(self.file_paths)} patient files.")

#         self.real_size = len(self.file_paths)
#         self.virtual_size = self.real_size * 100

#     def _find_npz_files(self):
#         """Scans the directory for .npz files."""
#         paths = []
#         for file_name in sorted(os.listdir(self.root_dir)):
#             if file_name.endswith('.npz') and not file_name.startswith('.'):
#                 paths.append(os.path.join(self.root_dir, file_name))
#         return paths

#     def __len__(self):
#         return self.virtual_size

#     def __getitem__(self, idx):
#         real_idx = idx % self.real_size
#         file_path = self.file_paths[real_idx]
        
#         try:
#             data = np.load(file_path)
#             full_tensor = data['vol'].astype(np.float32)

#             # --- Select correct channels and create the difference map ---
#             # Index 1: baseline T1ce
#             # Index 5: follow-up T1ce
#             baseline_t1ce = full_tensor[1]
#             followup_t1ce = full_tensor[5]
            
#             difference_map = followup_t1ce - baseline_t1ce
            
#             two_channel_numpy = np.stack([baseline_t1ce, difference_map], axis=0)
            
#             two_channel_tensor = torch.from_numpy(two_channel_numpy).unsqueeze(0)
            
#             resized_tensor = F.interpolate(
#                 two_channel_tensor, 
#                 size=(self.resolution, self.resolution, self.resolution), 
#                 mode='trilinear', 
#                 align_corners=False
#             )
            
#             return resized_tensor.squeeze(0), self.class_id
        
#         except Exception as e:
#             print(f"Error loading data from file {file_path}: {e}")
#             return torch.zeros(2, self.resolution, self.resolution, self.resolution), self.class_id
# # #H-code
# # import os
# # import numpy as np
# # import torch
# # from torch.utils.data import Dataset
# # import torch.nn.functional as F
# # import math
# # import random
# # from networks import Generator

# # class LumiereDataset(Dataset):
# #     """
# #     Dataset for loading a 2-channel tensor (baseline T1ce + follow-up T1ce)
# #     from pre-processed .npz files.
# #     """
# #     def __init__(self, root_dir, target_resolution=64, class_id=0):
# #         self.root_dir = os.path.join(root_dir, str(class_id))
# #         self.resolution = target_resolution
# #         self.class_id = class_id
        
# #         print(f"Scanning for patient data in: {self.root_dir}...")
# #         self.file_paths = self._find_npz_files()
# #         if not self.file_paths:
# #             raise FileNotFoundError(f"No .npz files found for class {class_id} in {self.root_dir}")
# #         print(f"Found {len(self.file_paths)} patient files.")

# #         self.real_size = len(self.file_paths)
# #         self.virtual_size = self.real_size * 100

# #     def _find_npz_files(self):
# #         """Scans the directory for .npz files."""
# #         paths = []
# #         for file_name in sorted(os.listdir(self.root_dir)):
# #             if file_name.endswith('.npz') and not file_name.startswith('.'):
# #                 paths.append(os.path.join(self.root_dir, file_name))
# #         return paths

# #     def __len__(self):
# #         return self.virtual_size

# #     def __getitem__(self, idx):
# #         real_idx = idx % self.real_size
# #         file_path = self.file_paths[real_idx]
        
# #         try:
# #             # --- Load the 10-channel .npz file ---
# #             data = np.load(file_path)
# #             full_tensor = data['vol'].astype(np.float32)

# #             # --- Select and stack the 2 required channels ---
# #             # Index 1: baseline T1ce
# #             # Index 6: follow-up T1ce
# #             baseline_t1ce = full_tensor[1]
# #             followup_t1ce = full_tensor[6]
            
# #             two_channel_numpy = np.stack([baseline_t1ce, followup_t1ce], axis=0)
            
# #             # --- Preprocess and resize the 2-channel tensor ---
# #             two_channel_tensor = torch.from_numpy(two_channel_numpy)
            
# #             # Add a batch dimension for resizing
# #             two_channel_tensor = two_channel_tensor.unsqueeze(0)
            
# #             resized_tensor = F.interpolate(
# #                 two_channel_tensor, 
# #                 size=(self.resolution, self.resolution, self.resolution), 
# #                 mode='trilinear', 
# #                 align_corners=False
# #             )
            
# #             return resized_tensor.squeeze(0), self.class_id
        
# #         except Exception as e:
# #             print(f"Error loading data from file {file_path}: {e}")
# #             return torch.zeros(2, self.resolution, self.resolution, self.resolution), self.class_id
# # import os
# # import numpy as np
# # import torch
# # from torch.utils.data import Dataset
# # import torch.nn.functional as F

# # class LumiereDataset(Dataset):
# #     def __init__(self, root_dir, target_resolution=64, class_id=0):
# #         self.root_dir = os.path.join(root_dir, str(class_id))
# #         self.resolution = target_resolution
# #         self.class_id = class_id
        
# #         print(f"Scanning for patient data in: {self.root_dir}...")
# #         self.file_paths = self._find_npz_files()
# #         if not self.file_paths:
# #             raise FileNotFoundError(f"No .npz files found for class {class_id} in {self.root_dir}")
# #         print(f"Found {len(self.file_paths)} patients for class {class_id}.")
        
# #         # --- NEW: Data Multiplicity for Speed ---
# #         # We "trick" the dataloader into thinking our dataset is larger.
# #         # This creates more batches and keeps the GPU pipeline full.
# #         self.real_size = len(self.file_paths)
# #         self.virtual_size = self.real_size * 100

# #     def _find_npz_files(self):
# #         paths = []
# #         for file_name in sorted(os.listdir(self.root_dir)):
# #             if file_name.startswith('.') or not file_name.endswith('.npz'):
# #                 continue
# #             path = os.path.join(self.root_dir, file_name)
# #             paths.append(path)
# #         return paths

# #     def __len__(self):
# #         # Report the larger, virtual size
# #         return self.virtual_size

# #     def __getitem__(self, idx):
# #         # Use the modulo operator to loop over the smaller, real dataset
# #         real_idx = idx % self.real_size
# #         file_path = self.file_paths[real_idx]
        
# #         try:
# #             data = np.load(file_path)
# #             tensor_10ch = data['vol']
# #             single_channel_volume = tensor_10ch[1].astype(np.float32)

# #             min_val, max_val = np.min(single_channel_volume), np.max(single_channel_volume)
# #             if max_val - min_val > 0:
# #                 single_channel_volume = 2 * (single_channel_volume - min_val) / (max_val - min_val) - 1
            
# #             volume_tensor = torch.from_numpy(single_channel_volume).unsqueeze(0)
# #             volume_tensor = volume_tensor.unsqueeze(0)
            
# #             resized_tensor = F.interpolate(
# #                 volume_tensor, 
# #                 size=(self.resolution, self.resolution, self.resolution), 
# #                 mode='trilinear', 
# #                 align_corners=False
# #             )
            
# #             return resized_tensor.squeeze(0), self.class_id
        
# #         except Exception as e:
# #             print(f"Error loading data from file {file_path}: {e}")
# #             return torch.zeros(1, self.resolution, self.resolution, self.resolution), self.class_id