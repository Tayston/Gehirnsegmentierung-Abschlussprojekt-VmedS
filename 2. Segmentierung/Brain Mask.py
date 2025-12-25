import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, label

# -------------------------------------------------------------------------
# Funktion: Brain-Maske für einen Slice erstellen (Relative Threshold)
# -------------------------------------------------------------------------
def get_brain_mask(slice_img, threshold=0.05):
    """
    Erstellt eine binäre Maske für das Gehirn aus einem 2D-Slice.
    
    Parameters:
        slice_img : 2D numpy array
            Einzelner Slice des MRI
        threshold : float
            Bruchteil des Maximalwerts, der als Gehirngewebe gilt
    
    Returns:
        brain_mask_clean : 2D boolean array
            Maske, True = Gehirn, False = Hintergrund
    """
    # 1. Schwellenwert
    thresh_val = threshold * np.max(slice_img)
    mask = slice_img > thresh_val

    # 2. Löcher füllen
    mask_filled = binary_fill_holes(mask)

    # 3. Größte zusammenhängende Komponente behalten
    labels, num = label(mask_filled)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # Hintergrund ignorieren
    largest_label = sizes.argmax()
    brain_mask_clean = labels == largest_label

    return brain_mask_clean

# -------------------------------------------------------------------------
# Funktion: MRI Slice laden (korrekte Orientierung)
# -------------------------------------------------------------------------
def load_slice_corrected(path, name):
    """
    Lädt ein MRI, extrahiert den mittleren Slice und korrigiert die Orientierung.
    
    Returns:
        slice_img : 2D numpy array
    """
    print(f"\n--- Lade {name} ---")
    img = np.asarray(nib.load(path).dataobj)
    print(f"{name} Original Form:", img.shape)

    # Mittleren Slice auswählen (Z-Achse)
    z_idx = img.shape[2] // 2
    slice_img = img[:, :, z_idx].astype(float)

    # Transpose für korrekte X/Y-Darstellung in Matplotlib
    slice_img = np.transpose(slice_img)

    # Flip vertikal + horizontal, damit Gehirn korrekt oben ist
    slice_img = np.flipud(slice_img)
    slice_img = np.fliplr(slice_img)

    return slice_img

# -------------------------------------------------------------------------
# Parameter
# -------------------------------------------------------------------------
brain_threshold = 0.05          # relativer Schwellenwert
t1_upper_threshold = 1200       # Intensität oberhalb -> wahrscheinlich Schädel
flair_lower_threshold = 50      # Intensität unterhalb -> kein Gehirn

# -------------------------------------------------------------------------
# MRI Slices laden
# -------------------------------------------------------------------------
T1_slice = load_slice_corrected("data/pat13_reg_T1.nii", "T1")
FLAIR_slice = load_slice_corrected("data/pat13_reg_FLAIR.nii", "FLAIR")

# -------------------------------------------------------------------------
# Brain-Masken erstellen
# -------------------------------------------------------------------------
T1_mask = get_brain_mask(T1_slice, threshold=brain_threshold)
FLAIR_mask = get_brain_mask(FLAIR_slice, threshold=brain_threshold)

# Kombinierte Maske (OR)
brain_mask_combined = T1_mask | FLAIR_mask

# -------------------------------------------------------------------------
# Intensity-basiertes Skull-Stripping
# -------------------------------------------------------------------------
brain_mask_filtered = brain_mask_combined & (T1_slice < t1_upper_threshold)
brain_mask_filtered = brain_mask_filtered & (FLAIR_slice > flair_lower_threshold)
brain_mask_stripped = binary_fill_holes(brain_mask_filtered)

# -------------------------------------------------------------------------
# Maskierte Gehirnbilder erstellen
# -------------------------------------------------------------------------
T1_brain_stripped = T1_slice * brain_mask_stripped
FLAIR_brain_stripped = FLAIR_slice * brain_mask_stripped

# -------------------------------------------------------------------------
# Visualisierung: Original Slices + Skull-stripped Brain
# -------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(20,5))

# T1 Original
axs[0].imshow(T1_slice, cmap="gray", origin="lower")
axs[0].set_title("T1 Original")
axs[0].axis("off")

# FLAIR Original
axs[1].imshow(FLAIR_slice, cmap="gray", origin="lower")
axs[1].set_title("FLAIR Original")
axs[1].axis("off")

# Brain Mask
axs[2].imshow(T1_brain_stripped, cmap="gray", origin="lower")
axs[2].imshow(brain_mask_stripped, cmap="Reds", alpha=0.3, origin="lower")
axs[2].set_title("Brain Mask")
axs[2].axis("off")

plt.show()