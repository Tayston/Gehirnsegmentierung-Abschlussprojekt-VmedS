import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def process_modality(path, name, sigmas=[1.0, 3.0]):
    print(f"\n--- Loading {name} ---")
    img = np.transpose(np.asarray(nib.load(path).dataobj), (1, 0, 2))
    print(f"{name} shape:", img.shape)

    # Middle slice
    z_idx = img.shape[2] // 2
    print(f"{name} selected slice index:", z_idx)
    slice_img = img[:, :, z_idx].astype(float)

    # Compute derivatives
    dy, dx = np.gradient(slice_img)

    # Apply Gaussian smoothing for each sigma
    smoothed_images = [gaussian_filter(slice_img, sigma=s) for s in sigmas]

T1, T1_slice, T1_dx, T1_dy, T1_smoothed = process_modality("pat13_reg_T1.nii", "T1", sigmas=sigmas)
FLAIR, FLAIR_slice, FLAIR_dx, FLAIR_dy, FLAIR_smoothed = process_modality("pat13_reg_FLAIR.nii", "FLAIR", sigmas=sigmas)
IR, IR_slice, IR_dx, IR_dy, IR_smoothed = process_modality("pat13_reg_IR.nii", "IR", sigmas=sigmas)

print("\nAll modalities processed successfully with derivatives and Gaussian smoothing.")
