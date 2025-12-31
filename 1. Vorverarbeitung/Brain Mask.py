"""
BRAIN MASK

ZIEL:
Extraktion des intrakraniellen Volumens durch Entfernung extrakranieller Strukturen (Schädel, 
Meningen, Hintergrund) zur Vorbereitung der Gewebesegmentierung.

METHODIK:
1. Robuste Normalisierung: Skalierung der Intensitäten auf 0-255 mittels Perzentilen (1%/99%), 
   um Ausreißer zu eliminieren und patientenübergreifende Schwellenwerte zu ermöglichen.
2. Morphologische Verarbeitung: Kombination aus adaptivem Thresholding, Hole-Filling (für Ventrikel) 
   und Largest-Connected-Component-Analyse zur Artefaktbereinigung.
3. Geometrischer Constraint: Anwendung einer zentralen ROI, um Randartefakte (Hals, Augen) deterministisch 
   zu verwerfen.

INTEGRATION:
Die generierte binäre Maske dient als Filter für die anschließende Feature-Extraktion im K-Means-Modul.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, label

# ============================================================
# 1) Vorverarbeitung & Normalisierung
# ============================================================

def robust_minmax_to_uint8(vol, p_low=1, p_high=99):
    """
    Führt eine robuste 8-Bit-Skalierung mittels Perzentil-Clipping durch.
    Verhindert Kontrastkompression durch Intensitäts-Spikes und standardisiert 
    die Dynamik für nachfolgende Threshold-Operationen.
    """
    v = vol.astype(np.float32)
    lo = np.percentile(v, p_low)
    hi = np.percentile(v, p_high)
    
    if hi <= lo + 1e-6:
        return np.zeros_like(v, dtype=np.uint8)

    v = np.clip(v, lo, hi)
    v = (v - lo) / (hi - lo)
    return (v * 255.0).astype(np.uint8)

def load_and_orient_slice(path):
    """
    Lädt NIfTI-Volumen, normalisiert global und extrahiert den mittleren axialen Slice.
    Transformiert das Array in die radiologische Standardorientierung (L=R, Superior=Up).
    """
    print(f"--- Lade {path} ---")
    vol = nib.load(path).get_fdata()
    
    # Globale Normalisierung vor Slice-Extraktion für statistische Stabilität
    vol_uint8 = robust_minmax_to_uint8(vol)
    
    z_idx = vol_uint8.shape[2] // 2
    slice_img = vol_uint8[:, :, z_idx]
    
    # Orientierungskorrektur (Matrix -> Kartesisch)
    slice_img = np.transpose(slice_img)
    slice_img = np.fliplr(slice_img)
    slice_img = np.flipud(slice_img)
    
    return slice_img

# ============================================================
# 2) Kern-Logik: Brain Masking
# ============================================================

def get_brain_mask(slice_uint8, thr_rel=0.10):
    """
    Generiert binäre Gehirnmaske mittels morphologischer Pipeline.
    
    Ablauf:
    1. Thresholding (>10% max): Trennung Signal vs. Hintergrund.
    2. binary_fill_holes: Einschluss hypointenser Areale (z.B. Ventrikel in T1).
    3. Labeling: Selektion der größten Zusammenhangskomponente zur Elimination 
       isolierter Artefakte (z.B. Orbita, Rauschen).
    """
    thresh = thr_rel * float(np.max(slice_uint8))
    
    m = slice_uint8 > thresh
    m = binary_fill_holes(m)

    lbl, num = label(m)
    if num == 0:
        return np.zeros_like(slice_uint8, dtype=bool)

    # Identifikation der Hauptkomponente (Gehirn)
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0 # Hintergrund exkludieren
    return lbl == sizes.argmax()

def apply_center_constraint(mask, keep_frac=0.85):
    """
    Wendet eine zentrale ROI an, um periphere Artefakte (Meningen, Halsansatz) zu verwerfen.
    Nutzt die A-priori-Information der Bildzentrierung nach Registrierung.
    """
    h, w = mask.shape
    ch, cw = h // 2, w // 2
    hh, ww = int(h * keep_frac / 2), int(w * keep_frac / 2)

    m2 = np.zeros_like(mask, dtype=bool)
    m2[ch - hh: ch + hh, cw - ww: cw + ww] = True
    
    return mask & m2

# ============================================================
# 3) Ausführung: Pipeline für Pat 7 & 13
# ============================================================

def process_and_visualize_mask(t1_path, patient_id):
    """
    Wrapper für Lade-, Maskierungs- und Stripping-Prozess auf T1-Daten.
    """
    # 1. Laden & Normalisieren
    t1_slice = load_and_orient_slice(t1_path)
    
    # 2. Maskengenerierung (Roh + ROI)
    mask_raw = get_brain_mask(t1_slice, thr_rel=0.10)
    mask_final = apply_center_constraint(mask_raw, keep_frac=0.85)
    
    # 3. Applikation der Maske
    t1_stripped = t1_slice.copy()
    t1_stripped[~mask_final] = 0
    
    return t1_slice, mask_final, t1_stripped

# Pfade (relative Pfade zum 'data' Ordner vorausgesetzt)
pat7_t1, pat7_mask, pat7_stripped = process_and_visualize_mask("data/pat7_reg_T1.nii", 7)
pat13_t1, pat13_mask, pat13_stripped = process_and_visualize_mask("data/pat13_reg_T1.nii", 13)

# ============================================================
# 4) Visualisierung
# ============================================================

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Brain-Masking Pipeline (T1-basiert)", fontsize=16)

# Patient 7
axs[0, 0].imshow(pat7_t1, cmap="gray", origin="lower")
axs[0, 0].set_title("Pat 7: T1 (Normalized)")
axs[0, 1].imshow(pat7_t1, cmap="gray", origin="lower")
axs[0, 1].imshow(pat7_mask, cmap="Reds", alpha=0.5, origin="lower")
axs[0, 1].set_title("Pat 7: Mask Overlay")
axs[0, 2].imshow(pat7_stripped, cmap="gray", origin="lower")
axs[0, 2].set_title("Pat 7: Skull-Stripped Result")

# Patient 13
axs[1, 0].imshow(pat13_t1, cmap="gray", origin="lower")
axs[1, 0].set_title("Pat 13: T1 (Normalized)")
axs[1, 1].imshow(pat13_t1, cmap="gray", origin="lower")
axs[1, 1].imshow(pat13_mask, cmap="Reds", alpha=0.5, origin="lower")
axs[1, 1].set_title("Pat 13: Mask Overlay")
axs[1, 2].imshow(pat13_stripped, cmap="gray", origin="lower")
axs[1, 2].set_title("Pat 13: Skull-Stripped Result")

for ax in axs.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()