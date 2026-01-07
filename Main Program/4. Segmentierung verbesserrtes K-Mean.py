import numpy as np  
import nibabel as nib  
import matplotlib.pyplot as plt  
from scipy.ndimage import binary_fill_holes, label  

# ============================================================
# 1) Robust 0..255 Normierung (pro 3D-Volumen, deterministisch)
# ============================================================

def robust_minmax_to_uint8(vol, p_low=1, p_high=99): 
    """
    Robustes Min-Max-Scaling auf 0..255 (uint8).
    Wichtig für den K-Means Algorithmus, damit alle Modalitäten 
    gleich gewichtet sind.
    """
    v = vol.astype(np.float32)
    lo = np.percentile(v, p_low)
    hi = np.percentile(v, p_high)
    if hi <= lo + 1e-6:
        return np.zeros_like(v, dtype=np.uint8)

    v = np.clip(v, lo, hi)
    v = (v - lo) / (hi - lo)
    return (v * 255.0).astype(np.uint8)

def orient_slice(vol, z): 
    """
    Entnimmt einen Slice und korrigiert die Orientierung.
    Funktioniert sowohl für Rohdaten (float) als auch für normierte Daten (uint8).
    """
    s = np.transpose(vol[:, :, z])
    s = np.fliplr(s)
    s = np.flipud(s)
    return s

# ============================================================
# 2) Brain Mask + optionaler Center-Constraint
# ============================================================

def get_brain_mask(slice_uint8, thr_rel=0.10):  
    thresh = thr_rel * float(np.max(slice_uint8))
    m = slice_uint8 > thresh
    m = binary_fill_holes(m)

    lbl, num = label(m)
    if num == 0:
        return np.zeros_like(slice_uint8, dtype=bool)

    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    return lbl == sizes.argmax()

def apply_center_constraint(mask, keep_frac=0.85):  
    h, w = mask.shape
    ch, cw = h // 2, w // 2
    hh, ww = int(h * keep_frac / 2), int(w * keep_frac / 2)

    m2 = np.zeros_like(mask, dtype=bool)
    m2[ch - hh: ch + hh, cw - ww: cw + ww] = True
    return mask & m2

# ============================================================
# 3) K-Means (deterministisch)
# ============================================================

def kmeans_fixed_init(X, init_centers, max_iter=40):
    centers = init_centers.astype(np.float32).copy()
    K = centers.shape[0]

    labels = None
    for _ in range(max_iter):
        dist = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dist, axis=1)

        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for k in range(K):
            pts = X[labels == k]
            if pts.shape[0] > 0:
                centers[k] = pts.mean(axis=0)

    return labels, centers

def labels_to_segmap(labels, idx, shape):
    seg = np.zeros(shape, dtype=np.uint8)
    seg[idx[0], idx[1]] = (labels.astype(np.uint8) + 1)
    return seg

def create_rgb(seg):
    rgb = np.zeros(seg.shape + (3,), dtype=np.float32)
    rgb[seg == 1, 2] = 1.0  # CSF: blau
    rgb[seg == 2, 1] = 1.0  # GM: grün
    rgb[seg == 3, 0] = 1.0  # WM: rot
    return rgb

# ============================================================
# 4) Neue Startzentren: deterministisch aus Quantilen
# ============================================================

def compute_centers_from_quantiles(s_t1, s_fl, s_ir, mask,
                                  q_csf=0.15, q_gm=0.55, q_wm=0.90,
                                  band=10):
    t = s_t1[mask].astype(np.float32)
    f = s_fl[mask].astype(np.float32)
    r = s_ir[mask].astype(np.float32)

    if t.size < 50:
        return (
            np.array([20,  80,  80], dtype=np.float32),
            np.array([110, 110, 110], dtype=np.float32),
            np.array([235, 170, 170], dtype=np.float32)
        )

    t_csf = float(np.quantile(t, q_csf))
    t_gm  = float(np.quantile(t, q_gm))
    t_wm  = float(np.quantile(t, q_wm))

    csf_band = t <= (t_csf + band)
    gm_band  = (t >= (t_gm - band)) & (t <= (t_gm + band))
    wm_band  = t >= (t_wm - band)

    f_med = float(np.median(f))
    r_med = float(np.median(r))

    def band_median(arr, band_mask, fallback):
        return float(np.median(arr[band_mask])) if np.any(band_mask) else float(fallback)

    c_csf = np.array([t_csf, band_median(f, csf_band, f_med), band_median(r, csf_band, r_med)], dtype=np.float32)
    c_gm  = np.array([t_gm,  band_median(f, gm_band,  f_med), band_median(r, gm_band,  r_med)], dtype=np.float32)
    c_wm  = np.array([t_wm,  band_median(f, wm_band,  f_med), band_median(r, wm_band,  r_med)], dtype=np.float32)

    return c_csf, c_gm, c_wm

# ============================================================
# 5) Pipeline pro Patient
# ============================================================

def segment_patient_quantile_centers(t1_path, flair_path, ir_path, patient_id,
                                    z_mode="mid",
                                    p_low=1, p_high=99,
                                    thr_rel=0.10,
                                    use_center_constraint=True,
                                    keep_frac=0.85,
                                    q_csf=0.15, q_gm=0.55, q_wm=0.90,
                                    band=10,
                                    weights=(1.2, 1.2, 0.8)):
    
    print(f"\n=== Patient {patient_id} ===")

    # 1. Laden der RAW Daten (float)
    t1_raw = nib.load(t1_path).get_fdata()
    fl_raw = nib.load(flair_path).get_fdata()
    ir_raw = nib.load(ir_path).get_fdata()

    # 2. Erstellen der NORMALISIERTEN Daten (für K-Means Algorithmus)
    # Dies ist notwendig für den Algorithmus, aber schlecht für die Visualisierung von IR
    t1_8 = robust_minmax_to_uint8(t1_raw, p_low, p_high)
    fl_8 = robust_minmax_to_uint8(fl_raw, p_low, p_high)
    ir_8 = robust_minmax_to_uint8(ir_raw, p_low, p_high)

    depth = t1_8.shape[2]
    if z_mode == "mid":
        z = depth // 2
    else:
        z = int(depth * float(z_mode))

    # 3. Slices extrahieren:
    # A) Die NORMALISIERTEN Slices (für Berechnung/Maske)
    s_t1_norm = orient_slice(t1_8, z)
    s_fl_norm = orient_slice(fl_8, z)
    s_ir_norm = orient_slice(ir_8, z)
    
    # B) Die RAW Slices (für Visualisierung -> besserer Kontrast!)
    s_t1_viz = orient_slice(t1_raw, z)
    s_fl_viz = orient_slice(fl_raw, z)
    s_ir_viz = orient_slice(ir_raw, z)

    # Maske (basiert auf T1 normalisiert)
    mask = get_brain_mask(s_t1_norm, thr_rel=thr_rel)
    if use_center_constraint:
        mask = apply_center_constraint(mask, keep_frac=keep_frac)

    idx = np.where(mask)
    if idx[0].size == 0:
        raise RuntimeError("Maske ist leer. thr_rel reduzieren oder keep_frac erhöhen.")

    # Features (basieren auf NORMALISIERTEN Daten)
    X = np.stack([s_t1_norm[idx], s_fl_norm[idx], s_ir_norm[idx]], axis=1).astype(np.float32)

    w = np.array(weights, dtype=np.float32)
    Xw = X * w

    # Startzentren
    c_csf, c_gm, c_wm = compute_centers_from_quantiles(
        s_t1_norm, s_fl_norm, s_ir_norm, mask,
        q_csf=q_csf, q_gm=q_gm, q_wm=q_wm,
        band=band
    )

    init_centers = np.stack([c_csf * w, c_gm * w, c_wm * w], axis=0)
    labels, centers_w = kmeans_fixed_init(Xw, init_centers, max_iter=40)
    seg = labels_to_segmap(labels, idx, s_t1_norm.shape)
    centers_unweighted = centers_w / (w + 1e-12)

    return {
        "patient_id": patient_id,
        "z": z,
        "t1": s_t1_viz,      # RAW für Visualisierung
        "flair": s_fl_viz,   # RAW für Visualisierung
        "ir": s_ir_viz,      # RAW für Visualisierung
        "mask": mask,
        "seg_rgb": create_rgb(seg),
        "init_centers_unweighted": np.stack([c_csf, c_gm, c_wm], axis=0),
        "final_centers_unweighted": centers_unweighted
    }

# ============================================================
# 6) Execution & 2x6 Grid Visualisierung
# ============================================================

res7 = segment_patient_quantile_centers(
    "data/pat7_reg_T1.nii", "data/pat7_reg_FLAIR.nii", "data/pat7_reg_IR.nii", 
    patient_id=7, weights=(1.2, 1.2, 0.8)
)

res13 = segment_patient_quantile_centers(
    "data/pat13_reg_T1.nii", "data/pat13_reg_FLAIR.nii", "data/pat13_reg_IR.nii", 
    patient_id=13, weights=(1.2, 1.2, 0.8)
)

# 2x6 Grid: T1, FLAIR, IR, Mask, Seg, Stats
fig, axs = plt.subplots(2, 6, figsize=(30, 10))  
for r, res in enumerate([res7, res13]):
    
    # 1. T1 Original (Raw)
    axs[r, 0].imshow(res["t1"], cmap="gray", origin="lower")
    axs[r, 0].set_title(f"Pat{res['patient_id']} T1")
    axs[r, 0].axis("off")

    # 2. FLAIR Original (Raw)
    axs[r, 1].imshow(res["flair"], cmap="gray", origin="lower")
    axs[r, 1].set_title(f"Pat{res['patient_id']} FLAIR")
    axs[r, 1].axis("off")

    # 3. IR Original (Raw)
    axs[r, 2].imshow(res["ir"], cmap="gray", origin="lower")
    axs[r, 2].set_title(f"Pat{res['patient_id']} IR")
    axs[r, 2].axis("off")

    # 4. Brain Mask Overlay
    axs[r, 3].imshow(res["t1"], cmap="gray", origin="lower")
    axs[r, 3].imshow(res["mask"], cmap="Reds", alpha=0.35, origin="lower")
    axs[r, 3].set_title("Brain Mask")
    axs[r, 3].axis("off")

    # 5. Segmentierung
    axs[r, 4].imshow(res["t1"], cmap="gray", origin="lower")
    axs[r, 4].imshow(res["seg_rgb"], alpha=0.55, origin="lower")
    axs[r, 4].set_title("Segmentation")
    axs[r, 4].axis("off")

    # 6. Text (FIXED)
    axs[r, 5].axis("off")
    ic = res["init_centers_unweighted"]
    fc = res["final_centers_unweighted"]
    
    # We round the arrays to 1 decimal place so they look clean but don't crash
    txt = (
        "Init-Zentren\n"
        f"CSF: {np.round(ic[0], 1)}\nGM : {np.round(ic[1], 1)}\nWM : {np.round(ic[2], 1)}\n\n"
        "Final-Zentren\n"
        f"CSF: {np.round(fc[0], 1)}\nGM : {np.round(fc[1], 1)}\nWM : {np.round(fc[2], 1)}"
    )
    axs[r, 5].text(0.0, 0.5, txt, fontsize=10, va="center")

plt.tight_layout()  
plt.show()