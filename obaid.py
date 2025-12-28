# Obaid Elhakim
# Version 1: Code funktioniert für beide Patienten, nur pro Slice Segmentierung und keine Vergleiche
# Aktuell wird immer die mittleren Slice verwendet
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, label

# -------------------------------------------------------------------------
# Funktion: Erstellung der Gehirnmaske
# -------------------------------------------------------------------------
def get_brain_mask(slice_img, threshold=0.05):
    """
    Erzeugt eine binäre Maske, die das Gehirn vom Hintergrund trennt.
    """
    # 1. Schwellenwert berechnen (relativ zum Maximalwert des Bildes)
    thresh_val = threshold * np.max(slice_img)
    mask = slice_img > thresh_val
    
    # 2. Löcher innerhalb der Maske füllen (z.B. Ventrikel oder Rauschen)
    mask_filled = binary_fill_holes(mask)
    
    # 3. Zusammenhangsanalyse: Nur die größte Komponente behalten
    # Dies entfernt kleine helle Artefakte außerhalb des Schädels.
    labels, num = label(mask_filled)
    if num == 0: return np.zeros_like(slice_img, dtype=bool)
    
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0 # Hintergrund (Label 0) ignorieren
    largest_label = sizes.argmax()
    
    return labels == largest_label

# -------------------------------------------------------------------------
# Funktion: Laden und Korrektur der MRI-Daten
# -------------------------------------------------------------------------
def load_slice_corrected(path, name):
    """
    Lädt NIfTI-Daten, extrahiert den mittleren Slice und korrigiert die Achsen
    für eine anatomisch korrekte Anzeige in Matplotlib.
    """
    print(f"--- Lade {name} ---")
    img = np.asarray(nib.load(path).dataobj)
    
    # Extraktion des mittleren Slices auf der Z-Achse (Axialebene)
    z_idx = img.shape[2] // 2
    slice_img = img[:, :, z_idx].astype(float)
    
    # Korrektur der Orientierung (Matplotlib zeigt Arrays oft gespiegelt an)
    slice_img = np.transpose(slice_img)
    slice_img = np.flipud(slice_img)
    slice_img = np.fliplr(slice_img)
    
    return slice_img

# -------------------------------------------------------------------------
# UNIVERSAL PIPELINE: Vollständige Verarbeitung eines Patienten
# -------------------------------------------------------------------------
def process_patient_full_pipeline(t1_path, flair_path, ir_path, patient_id):
    """
    Zentraler Workflow: Laden -> Skull Stripping -> Clustering -> Gewebeklassifizierung
    """
    print(f"\n=== START PIPELINE: PATIENT {patient_id} ===")
    
    # 1. Daten laden (T1 für Anatomie, FLAIR für Kontrast, IR für Gewebedifferenzierung)
    T1_slice = load_slice_corrected(t1_path, f"T1_pat{patient_id}")
    FLAIR_slice = load_slice_corrected(flair_path, f"FLAIR_pat{patient_id}")
    R1_slice = load_slice_corrected(ir_path, f"IR_pat{patient_id}")

    # 2. Skull-Stripping (Gehirnextraktion)
    # Kombiniert T1 und FLAIR Masken, um eine robuste Maske zu erhalten
    T1_mask = get_brain_mask(T1_slice)
    FLAIR_mask = get_brain_mask(FLAIR_slice)
    brain_mask_combined = T1_mask | FLAIR_mask

    # Zusätzliche Filterung basierend auf Intensität (Schädelknochen entfernen)
    t1_upper_threshold = 1200
    flair_lower_threshold = 50
    brain_mask_filtered = brain_mask_combined & (T1_slice < t1_upper_threshold)
    brain_mask_filtered = brain_mask_filtered & (FLAIR_slice > flair_lower_threshold)
    brain_mask_stripped = binary_fill_holes(brain_mask_filtered)

    # 3. Feature-Extraktion für das Clustering
    # Wir betrachten nur Pixel innerhalb der Gehirnmaske
    brain_idx = np.where(brain_mask_stripped)
    if brain_idx[0].size == 0:
        raise RuntimeError(f"Brain-Maske für Patient {patient_id} ist leer!")

    # Erstellung eines 3D-Feature-Vektors pro Pixel [T1, FLAIR, IR]
    feat_T1 = T1_slice[brain_idx]
    feat_FLAIR = FLAIR_slice[brain_idx]
    feat_IR = R1_slice[brain_idx]
    features = np.stack([feat_T1, feat_FLAIR, feat_IR], axis=1).astype(float)

    # 4. Normalisierung (Z-Score)
    # Wichtig, da verschiedene MRI-Sequenzen unterschiedliche Helligkeitsskalen haben
    eps = 1e-6
    features_norm = features.copy()
    for k in range(3):
        mu = np.mean(features_norm[:, k])
        sigma = np.std(features_norm[:, k]) + eps
        features_norm[:, k] = (features_norm[:, k] - mu) / sigma

    # 5. K-Means Clustering (Manuelle Implementierung)
    K = 3 # Ziel: CSF, Graue Substanz (GM), Weiße Substanz (WM)
    rng = np.random.default_rng(0)
    # Zufällige Initialisierung der Zentren
    rand_idx = rng.choice(features_norm.shape[0], size=K, replace=False)
    centers = features_norm[rand_idx, :]

    for it in range(20):
        # Distanzberechnung jedes Pixels zu jedem Zentrum
        dists = np.linalg.norm(features_norm[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1) # Zuordnung zum nächsten Zentrum
        
        # Zentren neu berechnen als Mittelwert der zugeordneten Punkte
        new_centers = np.array([features_norm[labels == k].mean(axis=0) if np.any(labels == k) 
                               else features_norm[rng.integers(0, features_norm.shape[0])] for k in range(K)])
        
        # Abbruch bei Konvergenz
        if np.linalg.norm(new_centers - centers) < 1e-3: break
        centers = new_centers

    # 6. Automatische Label-Zuordnung
    # Wir sortieren die Cluster nach der T1-Intensität: CSF (dunkel) < GM < WM (hell)
    cluster_T1_means = sorted([(k, np.mean(feat_T1[labels == k])) for k in range(K)], key=lambda x: x[1])
    csf_label, gm_label, wm_label = cluster_T1_means[0][0], cluster_T1_means[1][0], cluster_T1_means[2][0]

    # 7. Erstellung des RGB-Segmentierungsbildes
    seg_rgb = np.zeros(T1_slice.shape + (3,), dtype=float)
    seg_rgb[brain_idx[0][labels == csf_label], brain_idx[1][labels == csf_label], 2] = 1.0 # CSF = Blau
    seg_rgb[brain_idx[0][labels == gm_label], brain_idx[1][labels == gm_label], 1] = 1.0   # GM = Grün
    seg_rgb[brain_idx[0][labels == wm_label], brain_idx[1][labels == wm_label], 0] = 1.0   # WM = Rot

    return {
        "id": patient_id, "t1": T1_slice, "flair": FLAIR_slice, 
        "r1": R1_slice, "seg": seg_rgb, "mask": brain_mask_stripped,
        "t1_stripped": T1_slice * brain_mask_stripped
    }

# -------------------------------------------------------------------------
# HAUPTPROGRAMM / AUSFÜHRUNG
# -------------------------------------------------------------------------
# Verarbeitung beider Patienten über die Pipeline
results_pat7 = process_patient_full_pipeline("data/pat7_reg_T1.nii", "data/pat7_reg_FLAIR.nii", "data/pat7_reg_IR.nii", 7)
results_pat13 = process_patient_full_pipeline("data/pat13_reg_T1.nii", "data/pat13_reg_FLAIR.nii", "data/pat13_reg_IR.nii", 13)

# -------------------------------------------------------------------------
# VISUALISIERUNG: 2 Zeilen (Patienten) x 5 Spalten (Ergebnisse)
# -------------------------------------------------------------------------
fig, axs = plt.subplots(2, 5, figsize=(22, 10))
fig.suptitle("Abschlussprojekt: Multimodale Gehirnsegmentierung", fontsize=20)

for row, res in enumerate([results_pat7, results_pat13]):
    # Spalte 1-3: Die drei Eingangsmodalitäten
    axs[row, 0].imshow(res["t1"], cmap="gray", origin="lower")
    axs[row, 0].set_title(f"Pat{res['id']}: T1 (Anatomie)")
    axs[row, 1].imshow(res["flair"], cmap="gray", origin="lower")
    axs[row, 1].set_title(f"Pat{res['id']}: FLAIR")
    axs[row, 2].imshow(res["r1"], cmap="gray", origin="lower")
    axs[row, 2].set_title(f"Pat{res['id']}: IR")

    # Spalte 4: Visualisierung der Gehirnmaske (Skull-Stripping)
    axs[row, 3].imshow(res["t1_stripped"], cmap="gray", origin="lower")
    # Maske als rotes Overlay anzeigen
    axs[row, 3].imshow(res["mask"], cmap="Reds", alpha=0.3, origin="lower")
    axs[row, 3].set_title(f"Pat{res['id']}: Brain Mask Overlay")

    # Spalte 5: Farbiges Ergebnis der Segmentierung
    axs[row, 4].imshow(res["t1"], cmap="gray", origin="lower")
    axs[row, 4].imshow(res["seg"], alpha=0.6, origin="lower")
    axs[row, 4].set_title(f"Pat{res['id']}: Segmentiert (CSF/GM/WM)")

    # Achsen für alle Subplots ausblenden
    for ax in axs[row]: ax.axis("off")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()