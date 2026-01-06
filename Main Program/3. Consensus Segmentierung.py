import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, label
from scipy import stats  # Erforderlich für die statistische Mode-Berechnung (Mehrheitsentscheid)

# -------------------------------------------------------------------------
# MODUL 1: Vorverarbeitung & Maskierung
# -------------------------------------------------------------------------
def get_brain_mask(slice_img, threshold=0.05):
    """
    Erstellt eine binäre Maske zur Extraktion des intrakraniellen Volumens (Skull-Stripping).
    
    Verfahren:
    1. Adaptives Thresholding: Trennung von Signal und Hintergrund basierend auf 5% der Maximalintensität.
    2. Morphologische Operationen: 'binary_fill_holes' integriert hypointense Areale (z.B. Ventrikel),
       die sonst fälschlicherweise maskiert würden.
    3. Komponentenanalyse: Selektion des größten zusammenhängenden Volumens zur Elimination
       extrakranieller Artefakte.
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
# MODUL 2: Clustering-Logik
# -------------------------------------------------------------------------
def run_kmeans_segmentation(merkmale, merkmale_norm, pixel_indizes, shape):
    """
    Kernfunktion für das unüberwachte Lernen mittels K-Means.
    
    Parameter:
    - K=3: Zielklassen sind Liquor (CSF), Graue Substanz (GM), Weiße Substanz (WM).
    
    Funktionalität:
    Nach dem Clustering erfolgt eine deterministische Zuordnung der zufälligen Cluster-IDs
    zu anatomischen Gewebeklassen. Dies geschieht durch Sortierung der mittleren 
    T1-Signalintensität jedes Clusters (T1-Gewichtung: CSF dunkel < GM mittel < WM hell).
    """
    K = 3
    rng = np.random.default_rng()
    # Zufällige Initialisierung der Clusterzentren im normalisierten Feature-Raum
    zentren = merkmale_norm[rng.choice(merkmale_norm.shape[0], K, replace=False)]
    
    for _ in range(15): # Iterative Optimierung der Clusterzentren
        distanzen = np.linalg.norm(merkmale_norm[:, None, :] - zentren[None, :, :], axis=2)
        labels = np.argmin(distanzen, axis=1)
        zentren = np.array([merkmale_norm[labels == k].mean(axis=0) if np.any(labels == k) 
                            else merkmale_norm[rng.choice(merkmale_norm.shape[0])] for k in range(K)])
    
    # Automatische anatomische Label-Zuordnung basierend auf T1-Intensität
    t1_mittelwerte = [np.mean(merkmale[labels == k, 0]) for k in range(K)]
    sortierung = np.argsort(t1_mittelwerte) # Reihenfolge: CSF, GM, WM
    
    # Konstruktion der 2D-Segmentierungskarte (1=CSF, 2=GM, 3=WM)
    seg_map = np.zeros(shape)
    for ziel_wert, cluster_idx in enumerate(sortierung, 1):
        seg_map[pixel_indizes[0][labels == cluster_idx], pixel_indizes[1][labels == cluster_idx]] = ziel_wert
    return seg_map

# -------------------------------------------------------------------------
# MODUL 3: Volumetrische Pipeline (Consensus Voting)
# -------------------------------------------------------------------------
def process_patient_full_pipeline(t1_path, flair_path, ir_path, patient_id):
    """
    Zentraler Workflow: Vergleichende Analyse zwischen Single-Slice- und Multi-Slice-Ansatz.
    """
    print(f"\n=== VERARBEITUNG PATIENT {patient_id} ===")
    
    # 1. Laden der vollständigen 3D-Volumina (Voxel-Daten)
    t1_vol = nib.load(t1_path).get_fdata()
    flair_vol = nib.load(flair_path).get_fdata()
    ir_vol = nib.load(ir_path).get_fdata()
    
    tiefe = t1_vol.shape[2]
    mid_z = tiefe // 2 # Referenz-Index für den mittleren Slice
    
    # ---------------------------------------------------------------------
    # A) SINGLE-SLICE SEGMENTIERUNG (Baseline-Verfahren)
    # ---------------------------------------------------------------------
    def segment_single_slice(z):
        # Orientierungskorrektur in radiologische Ansicht
        s_t1 = np.flipud(np.fliplr(np.transpose(t1_vol[:, :, z])))
        s_flair = np.flipud(np.fliplr(np.transpose(flair_vol[:, :, z])))
        s_ir = np.flipud(np.fliplr(np.transpose(ir_vol[:, :, z])))
        
        # Maskierung und Feature-Extraktion
        mask = get_brain_mask(s_t1) & (s_t1 < 1200) 
        idx = np.where(mask)
        merkmale = np.stack([s_t1[idx], s_flair[idx], s_ir[idx]], axis=1)
        
        # Z-Score Normalisierung zur Standardisierung der Modalitäten
        merkmale_norm = (merkmale - np.mean(merkmale, axis=0)) / (np.std(merkmale, axis=0) + 1e-6)
        
        return run_kmeans_segmentation(merkmale, merkmale_norm, idx, s_t1.shape), s_t1, s_flair, s_ir

    # Berechnung des Baseline-Ergebnisses (nur mittlerer Slice)
    seg_single, t1_mid, flair_mid, ir_mid = segment_single_slice(mid_z)

    # ---------------------------------------------------------------------
    # B) VOLUMEN-VERGLEICH (45% - 55% Intervall)
    # ---------------------------------------------------------------------
    # Definition des zentralen Bereichs für die Consensus-Analyse
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

    # KONSENSUS-LOGIK:
    # Stapelung der segmentierten Slices zu einem 3D-Array.
    # Berechnung des Modalwerts (häufigstes Label) entlang der Z-Achse für jeden Voxel.
    # Dies eliminiert stochastisches Rauschen, da Gewebe über benachbarte Slices hinweg stabil ist.
    konsens_stack = np.stack(stapel, axis=0)
    final_labels, _ = stats.mode(konsens_stack, axis=0, keepdims=True)
    seg_consensus = final_labels[0]

    # 4. Hilfsfunktion zur RGB-Konvertierung für Visualisierung
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
# EXECUTION & VISUALISIERUNG
# -------------------------------------------------------------------------
results_pat7 = process_patient_full_pipeline("data/pat7_reg_T1.nii", "data/pat7_reg_FLAIR.nii", "data/pat7_reg_IR.nii", 7)
results_pat13 = process_patient_full_pipeline("data/pat13_reg_T1.nii", "data/pat13_reg_FLAIR.nii", "data/pat13_reg_IR.nii", 13)

# Grid-Darstellung: Vergleich der Methoden
fig, axs = plt.subplots(2, 5, figsize=(25, 12))
fig.suptitle("MRI Analyse: Vergleich Single-Slice vs. 3D-Volumen-Konsens", fontsize=22)

for row, res in enumerate([results_pat7, results_pat13]):
    # Spalte 1-3: Darstellung der Original-Sequenzen
    axs[row, 0].imshow(res["t1"], cmap="gray", origin="lower"); axs[row, 0].set_title(f"Pat{res['id']}: T1 Original"); axs[row, 0].axis("off")
    axs[row, 1].imshow(res["flair"], cmap="gray", origin="lower"); axs[row, 1].set_title(f"Pat{res['id']}: FLAIR Original"); axs[row, 1].axis("off")
    axs[row, 2].imshow(res["ir"], cmap="gray", origin="lower"); axs[row, 2].set_title(f"Pat{res['id']}: IR Original"); axs[row, 2].axis("off")
    
    # Spalte 4: Baseline-Ergebnis (Single-Slice)
    axs[row, 3].imshow(res["t1"], cmap="gray", origin="lower")
    axs[row, 3].imshow(res["seg_single"], alpha=0.5, origin="lower")
    axs[row, 3].set_title("Single-Slice Segmentierung"); axs[row, 3].axis("off")
    
    # Spalte 5: Optimiertes Ergebnis (Consensus Voting)
    axs[row, 4].imshow(res["t1"], cmap="gray", origin="lower")
    axs[row, 4].imshow(res["seg_consensus"], alpha=0.5, origin="lower")
    axs[row, 4].set_title(f"Consensus ({res['anzahl']} Slices)"); axs[row, 4].axis("off")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()