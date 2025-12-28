import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, label

# -------------------------------------------------------------------------
# Funktion: Brain-Maske für einen Slice erstellen (UNVERÄNDERT)
# -------------------------------------------------------------------------
def get_brain_mask(slice_img, threshold=0.05):
    """
    Erstellt eine binäre Maske für das Gehirn aus einem 2D-Slice.
    """
    thresh_val = threshold * np.max(slice_img)
    mask = slice_img > thresh_val
    mask_filled = binary_fill_holes(mask)
    labels, num = label(mask_filled)
    if num == 0: return np.zeros_like(slice_img, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  
    largest_label = sizes.argmax()
    brain_mask_clean = labels == largest_label
    return brain_mask_clean

# -------------------------------------------------------------------------
# Funktion: MRI Slice laden (UNVERÄNDERT)
# -------------------------------------------------------------------------
def load_slice_corrected(path, name):
    """
    Lädt ein MRI, extrahiert den mittleren Slice und korrigiert die Orientierung.
    """
    print(f"\n--- Lade {name} ---")
    img = np.asarray(nib.load(path).dataobj)
    z_idx = img.shape[2] // 2
    slice_img = img[:, :, z_idx].astype(float)
    slice_img = np.transpose(slice_img)
    slice_img = np.flipud(slice_img)
    slice_img = np.fliplr(slice_img)
    return slice_img

# -------------------------------------------------------------------------
# NEUE FUNKTION: Universelles Brain Masking & Skull Stripping
# -------------------------------------------------------------------------
def process_patient_brain_mask(t1_path, flair_path, patient_id, 
                               brain_thresh=0.05, t1_upper=1200, flair_lower=50):
    """
    Führt den gesamten Maskierungsprozess für einen Patienten durch.
    Gibt ein Dictionary mit allen relevanten Bildern/Masken zurück.
    """
    # 1. Laden
    t1 = load_slice_corrected(t1_path, f"Pat{patient_id} T1")
    flair = load_slice_corrected(flair_path, f"Pat{patient_id} FLAIR")

    # 2. Maskierung
    t1_m = get_brain_mask(t1, threshold=brain_thresh)
    flair_m = get_brain_mask(flair, threshold=brain_thresh)
    combined = t1_m | flair_m

    # 3. Stripping
    filtered = combined & (t1 < t1_upper) & (flair > flair_lower)
    stripped_mask = binary_fill_holes(filtered)
    
    # 4. Resultat
    t1_stripped = t1 * stripped_mask
    
    return {
        "t1": t1,
        "flair": flair,
        "mask": stripped_mask,
        "t1_stripped": t1_stripped,
        "id": patient_id
    }

# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------

# Patienten-Daten verarbeiten
pat7_results = process_patient_brain_mask("data/pat7_reg_T1.nii", "data/pat7_reg_FLAIR.nii", 7)
pat13_results = process_patient_brain_mask("data/pat13_reg_T1.nii", "data/pat13_reg_FLAIR.nii", 13)

# -------------------------------------------------------------------------
# Visualisierung: Vergleich beider Patienten
# -------------------------------------------------------------------------
fig, axs = plt.subplots(2, 3, figsize=(20, 10))

for i, res in enumerate([pat7_results, pat13_results]):
    # Original T1
    axs[i, 0].imshow(res["t1"], cmap="gray", origin="lower")
    axs[i, 0].set_title(f"Pat{res['id']}: T1 Original")
    axs[i, 0].axis("off")

    # Original FLAIR
    axs[i, 1].imshow(res["flair"], cmap="gray", origin="lower")
    axs[i, 1].set_title(f"Pat{res['id']}: FLAIR Original")
    axs[i, 1].axis("off")

    # Brain Mask Overlay
    axs[i, 2].imshow(res["t1_stripped"], cmap="gray", origin="lower")
    axs[i, 2].imshow(res["mask"], cmap="Reds", alpha=0.3, origin="lower")
    axs[i, 2].set_title(f"Pat{res['id']}: Brain Mask Overlay")
    axs[i, 2].axis("off")

plt.tight_layout()
plt.show()