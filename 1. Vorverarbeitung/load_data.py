import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# -------------------------------------------------------------------------
# Helper function to load, compute derivatives, and apply Gaussian smoothing
# -------------------------------------------------------------------------
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