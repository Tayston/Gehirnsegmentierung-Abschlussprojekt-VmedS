# Version 2: Erste Versuch, Vergleiche zwischen Slices zu implementieren, um Ergbenisse zu verbessern
# Aktuell werden immer die mittlere 45-55% der Slices miteinander verglichen
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, label
from scipy import stats  # Erforderlich für die statistische Mode-Berechnung (Mehrheitsentscheid)

# -------------------------------------------------------------------------
# Funktion: get_brain_mask (Skull-Stripping)
# -------------------------------------------------------------------------
def get_brain_mask(slice_img, threshold=0.05):
    """
    Erzeugt eine binäre Maske zur Trennung von Hirngewebe und Hintergrund.
    Nutzt adaptive Schwellenwertbildung und morphologische Operationen.
    """
    # 1. Schwellenwert relativ zur maximalen Intensität des Slices festlegen
    thresh_val = threshold * np.max(slice_img)
    mask = slice_img > thresh_val
    
    # 2. Füllen von Löchern innerhalb der Maske (z.B. dunkle Ventrikelbereiche)
    mask_filled = binary_fill_holes(mask)
    
    # 3. Extraktion der größten zusammenhängenden Region (entfernt Rauschen/Schädelreste)
    labels, num = label(mask_filled)
    if num == 0: return np.zeros_like(slice_img, dtype=bool)
    
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0 # Hintergrund ignorieren
    largest_label = sizes.argmax()
    
    return labels == largest_label

# -------------------------------------------------------------------------
# Hilfsfunktion: K-Means Kern-Logik
# -------------------------------------------------------------------------
def run_kmeans_segmentation(merkmale, merkmale_norm, pixel_indizes, shape,
                            weights=(1.5, 1.5, 0.7), init="kmeans++", max_iter=20):
    """
    K-Means (K=3) mit
    (1) Initialisierung: 'kmeans++' oder 'farthest'
    (2) Feature-Gewichtung: T1/FLAIR stärker als IR
    (3) Cluster-Sortierung nach T1-Helligkeit (CSF < GM < WM)
    """
    K = 3
    rng = np.random.default_rng()

    X = merkmale_norm.astype(np.float32)

    # ----------------------------
    # (A) Feature-Gewichtung
    # ----------------------------
    w = np.array(weights, dtype=np.float32)  # (T1, FLAIR, IR)
    Xw = X * w  # gewichteter Merkmalsraum für Distanzrechnung

    # ----------------------------
    # (B) Initialisierung: Zentren weit auseinander
    # ----------------------------
    n = Xw.shape[0]
    if n < K:
        # Notfall: zu wenige Punkte
        seg_map = np.zeros(shape)
        return seg_map

    def init_farthest_points(Xw, K):
        # Start: zufällig ein Punkt
        centers = np.empty((K, Xw.shape[1]), dtype=np.float32)
        idx0 = rng.integers(0, Xw.shape[0])
        centers[0] = Xw[idx0]

        # immer den Punkt wählen, der am weitesten zum nächsten Center ist
        dist2 = np.sum((Xw - centers[0]) ** 2, axis=1)
        for k in range(1, K):
            idx = np.argmax(dist2)
            centers[k] = Xw[idx]
            dist2 = np.minimum(dist2, np.sum((Xw - centers[k]) ** 2, axis=1))
        return centers

    def init_kmeans_pp(Xw, K):
        centers = np.empty((K, Xw.shape[1]), dtype=np.float32)
        idx0 = rng.integers(0, Xw.shape[0])
        print(f"[KMeans INIT] first_center_index={idx0}")
        centers[0] = Xw[idx0]

        dist2 = np.sum((Xw - centers[0]) ** 2, axis=1)
        for k in range(1, K):
            probs = dist2 / (dist2.sum() + 1e-12)
            idx = rng.choice(Xw.shape[0], p=probs)
            centers[k] = Xw[idx]
            dist2 = np.minimum(dist2, np.sum((Xw - centers[k]) ** 2, axis=1))
        return centers

    if init == "farthest":
        zentren = init_farthest_points(Xw, K)
    else:
        zentren = init_kmeans_pp(Xw, K)  # default: kmeans++

    # ----------------------------
    # (C) K-Means Iterationen (im gewichteten Raum)
    # ----------------------------
    labels = None
    for _ in range(max_iter):
        distanzen = np.linalg.norm(Xw[:, None, :] - zentren[None, :, :], axis=2)
        new_labels = np.argmin(distanzen, axis=1)

        # Konvergenzcheck
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Zentren aktualisieren (wichtig: ebenfalls im gewichteten Raum!)
        for k in range(K):
            pts = Xw[labels == k]
            if pts.shape[0] > 0:
                zentren[k] = pts.mean(axis=0)
            else:
                # leeres Cluster: re-seed auf weit entfernten Punkt
                zentren[k] = Xw[rng.integers(0, n)]

    # ----------------------------
    # (D) Cluster-Sortierung anatomisch (CSF < GM < WM) anhand T1 (ungewichtet!)
    # ----------------------------
    # labels beziehen sich auf Xw, aber die "Interpretation" machst du auf merkmale[:,0] = T1
    t1_mittelwerte = [np.mean(merkmale[labels == k, 0]) if np.any(labels == k) else np.inf for k in range(K)]
    sortierung = np.argsort(t1_mittelwerte)  # CSF, GM, WM

    seg_map = np.zeros(shape, dtype=np.uint8)
    for ziel_wert, cluster_idx in enumerate(sortierung, 1):
        seg_map[pixel_indizes[0][labels == cluster_idx],
                pixel_indizes[1][labels == cluster_idx]] = ziel_wert

    return seg_map


# -------------------------------------------------------------------------
# UNIVERSAL PIPELINE: Multimodale 3D-Volumen-Segmentierung
# -------------------------------------------------------------------------
def process_patient_full_pipeline(t1_path, flair_path, ir_path, patient_id):
    """
    Zentraler Workflow: Lädt 3D-Daten, berechnet eine Single-Slice-Segmentierung 
    sowie eine robustere Consensus-Segmentierung über den 45%-55% Bereich.
    """
    print(f"\n=== VERARBEITUNG PATIENT {patient_id} ===")
    
    # 1. Laden der NIfTI-Volumina
    t1_vol = nib.load(t1_path).get_fdata()
    flair_vol = nib.load(flair_path).get_fdata()
    ir_vol = nib.load(ir_path).get_fdata()
    
    tiefe = t1_vol.shape[2]
    mid_z = tiefe // 2 # Der exakte mittlere Slice als Referenz
    
    # 2. SINGLE-SLICE SEGMENTIERUNG (Klassischer Ansatz)
    def segment_single_slice(z):
        # Orientierung korrigieren
        s_t1 = np.flipud(np.fliplr(np.transpose(t1_vol[:, :, z])))
        s_flair = np.flipud(np.fliplr(np.transpose(flair_vol[:, :, z])))
        s_ir = np.flipud(np.fliplr(np.transpose(ir_vol[:, :, z])))
        
        mask = get_brain_mask(s_t1) & (s_t1 < 1200) # Grobes Skull-Stripping
        idx = np.where(mask)
        merkmale = np.stack([s_t1[idx], s_flair[idx], s_ir[idx]], axis=1)
        merkmale_norm = (merkmale - np.mean(merkmale, axis=0)) / (np.std(merkmale, axis=0) + 1e-6)
        
        return run_kmeans_segmentation(merkmale, merkmale_norm, idx, s_t1.shape), s_t1, s_flair, s_ir

    # Berechne Ergebnisse für den mittleren Slice
    seg_single, t1_mid, flair_mid, ir_mid = segment_single_slice(mid_z)

    # 3. VOLUMEN-VERGLEICH (45% - 55% Bereich)
    start_z, ende_z = int(tiefe * 0.45), int(tiefe * 0.55)
    stapel = []
    print(f"Vergleiche Slices {start_z} bis {ende_z} für Konsens-Entscheidung...")

    for z in range(start_z, ende_z):
        s_t1 = np.flipud(np.fliplr(np.transpose(t1_vol[:, :, z])))
        s_flair = np.flipud(np.fliplr(np.transpose(flair_vol[:, :, z])))
        s_ir = np.flipud(np.fliplr(np.transpose(ir_vol[:, :, z])))
        
        mask = get_brain_mask(s_t1) & (s_t1 < 1200)
        idx = np.where(mask)
        if idx[0].size == 0: continue
        
        m = np.stack([s_t1[idx], s_flair[idx], s_ir[idx]], axis=1)
        m_n = (m - np.mean(m, axis=0)) / (np.std(m, axis=0) + 1e-6)
        stapel.append(run_kmeans_segmentation(m, m_n, idx, s_t1.shape))

    # Konsens-Berechnung via Mehrheitsentscheid (Mode)
    konsens_stack = np.stack(stapel, axis=0)
    final_labels, _ = stats.mode(konsens_stack, axis=0, keepdims=True)
    seg_consensus = final_labels[0]

    # 4. Umwandlung in RGB-Bilder für die Anzeige
    def create_rgb(label_map):
        rgb = np.zeros(label_map.shape + (3,))
        rgb[label_map == 1, 2] = 1.0 # CSF = Blau
        rgb[label_map == 2, 1] = 1.0 # GM = Grün
        rgb[label_map == 3, 0] = 1.0 # WM = Rot
        return rgb

    return {
        "id": patient_id, "t1": t1_mid, "flair": flair_mid, "ir": ir_mid,
        "seg_single": create_rgb(seg_single),
        "seg_consensus": create_rgb(seg_consensus),
        "anzahl": len(stapel)
    }

# -------------------------------------------------------------------------
# EXECUTION & VISUALISIERUNG (2x5 Grid)
# -------------------------------------------------------------------------
results_pat7 = process_patient_full_pipeline("pat7_reg_T1.nii.gz", "pat7_reg_FLAIR.nii.gz", "pat7_reg_IR.nii.gz", 7)
results_pat13 = process_patient_full_pipeline("pat13_reg_T1.nii.gz", "pat13_reg_FLAIR.nii.gz", "pat13_reg_IR.nii.gz", 13)

fig, axs = plt.subplots(2, 5, figsize=(25, 12))
fig.suptitle("MRI Analyse: Vergleich Single-Slice vs. 3D-Volumen-Konsens", fontsize=22)

for row, res in enumerate([results_pat7, results_pat13]):
    # Spalte 1-3: Original Scans (T1, FLAIR, IR)
    axs[row, 0].imshow(res["t1"], cmap="gray", origin="lower"); axs[row, 0].set_title(f"Pat{res['id']}: T1 Original"); axs[row, 0].axis("off")
    axs[row, 1].imshow(res["flair"], cmap="gray", origin="lower"); axs[row, 1].set_title(f"Pat{res['id']}: FLAIR Original"); axs[row, 1].axis("off")
    axs[row, 2].imshow(res["ir"], cmap="gray", origin="lower"); axs[row, 2].set_title(f"Pat{res['id']}: IR Original"); axs[row, 2].axis("off")
    
    # Spalte 4: Herkömmliche Segmentierung (Nur mittlerer Slice)
    axs[row, 3].imshow(res["t1"], cmap="gray", origin="lower")
    axs[row, 3].imshow(res["seg_single"], alpha=0.5, origin="lower")
    axs[row, 3].set_title("Single-Slice Segmentierung"); axs[row, 3].axis("off")
    
    # Spalte 5: Optimierte Segmentierung (Slice-Vergleich / Konsens)
    axs[row, 4].imshow(res["t1"], cmap="gray", origin="lower")
    axs[row, 4].imshow(res["seg_consensus"], alpha=0.5, origin="lower")
    axs[row, 4].set_title(f"Consensus ({res['anzahl']} Slices)"); axs[row, 4].axis("off")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

