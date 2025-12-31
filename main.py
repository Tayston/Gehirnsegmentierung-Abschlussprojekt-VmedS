import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, label

# ============================================================
# 1) Robuste Normalisierung & Vorverarbeitung
# ============================================================

def robust_minmax_to_uint8(vol, p_low=1, p_high=99):
    """
    Führt eine robuste Min-Max-Skalierung auf den Wertebereich [0, 255] (8-Bit) durch.
    
    Hintergrund:
    MRT-Intensitätswerte sind nicht standardisiert (im Gegensatz zu Hounsfield-Einheiten im CT).
    Zudem enthalten MRT-Rohdaten oft extreme Ausreißer (z.B. helle Artefakte), die eine 
    klassische Min-Max-Skalierung verzerren würden.
    
    Vorgehen:
    1. Berechnung der Perzentile (z.B. 1. und 99.), um Ausreißer zu ignorieren.
    2. Clipping der Werte außerhalb dieses Bereichs (Sättigung).
    3. Lineare Transformation des verbleibenden Wertebereichs auf [0, 255].
    
    Parameter:
        vol: Das 3D-Volumen (Numpy Array).
        p_low, p_high: Untere und obere Perzentilgrenzen.
    
    Rückgabe:
        Skaliertes Volumen als uint8 (speichereffizient und standardisiert).
    """
    v = vol.astype(np.float32)  # Konvertierung für präzise Berechnungen
    lo = np.percentile(v, p_low)   # Untere Robustheitsgrenze
    hi = np.percentile(v, p_high)  # Obere Robustheitsgrenze

    # Sicherheitscheck: Verhindert Division durch Null bei flachen/leeren Bildern
    if hi <= lo + 1e-6:
        return np.zeros_like(v, dtype=np.uint8)

    # Clipping: Werte < lo werden zu lo, Werte > hi werden zu hi
    v = np.clip(v, lo, hi)
    
    # Lineare Skalierung auf [0, 1]
    v = (v - lo) / (hi - lo)
    
    # Skalierung auf [0, 255] und Konvertierung in Ganzzahlen
    return (v * 255.0).astype(np.uint8)

def orient_slice(vol_uint8, z):
    """
    Extrahiert einen axialen Schnitt (Slice) und korrigiert dessen Orientierung 
    für die radiologische Darstellung.
    
    Anatomischer Bezug:
    In Python-Bibliotheken (wie Matplotlib) ist der Koordinatenursprung oft oben links.
    Radiologische Bilder werden jedoch konventionell so betrachtet, als würde man von 
    den Füßen des Patienten nach oben schauen (Links im Bild = Rechts am Patienten).
    
    Vorgehen:
    1. Transposition: Vertauscht X- und Y-Achsen.
    2. Flip LR / Flip UD: Korrigiert Spiegelungen, sodass die Nase oben und
       die linke Gehirnhälfte rechts im Bild erscheint (radiologische Konvention).
    """
    s = np.transpose(vol_uint8[:, :, z])
    s = np.fliplr(s)
    s = np.flipud(s)
    return s

# ============================================================
# 2) Brain Mask (Skull Stripping) & ROI-Einschränkung
# ============================================================

def get_brain_mask(slice_uint8, thr_rel=0.10):
    """
    Erstellt eine binäre Maske zur Trennung von Gehirngewebe (Intrakraniell) 
    und Nicht-Gehirn (Schädelknochen, Kopfhaut, Hintergrund).
    
    Verfahren:
    1. Schwellenwertverfahren (Thresholding): Da der Hintergrund im MRT fast schwarz ist,
       nutzen wir einen relativen Schwellenwert (z.B. 10% der maximalen Intensität),
       um Gewebe vom Hintergrund zu trennen.
    2. Morphologische Operation (Lochfüllung): Anatomische Hohlräume wie die Ventrikel
       (mit Liquor gefüllt) können dunkel erscheinen. 'binary_fill_holes' stellt sicher,
       dass diese als Teil des Gehirns erkannt werden.
    3. Zusammenhangsanalyse (Connected Components): Um Artefakte außerhalb des Kopfes 
       zu entfernen, behalten wir nur das größte zusammenhängende Objekt (das Gehirn).
    
    Rückgabe:
        Binäre Maske (True = Gehirn, False = Hintergrund).
    """
    # Berechnung des absoluten Schwellenwerts basierend auf der Bilddynamik
    thresh = thr_rel * float(np.max(slice_uint8))
    
    # Binarisierung
    m = slice_uint8 > thresh
    
    # Schließt Löcher innerhalb des Gehirns (z.B. dunkle Ventrikel in T1)
    m = binary_fill_holes(m)

    # Identifikation aller getrennten Objekte im Bild
    lbl, num = label(m)
    if num == 0:
        return np.zeros_like(slice_uint8, dtype=bool)

    # Zählen der Pixel pro Objekt
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0  # Hintergrund (Label 0) darf nicht gewählt werden
    
    # Erstellen der Maske nur für das größte Objekt (das Gehirn)
    return lbl == sizes.argmax()

def apply_center_constraint(mask, keep_frac=0.85):
    """
    Wendet eine geometrische Einschränkung (Region of Interest, ROI) an.
    
    Zweck:
    Oft verbleiben bei einfachen Thresholding-Methoden Reste der Augen, der Nase 
    oder der Hirnhaut (Meningen) am Rand der Maske. Da das Gehirn zentral im 
    Bild liegt, können wir Ränder proaktiv verwerfen.
    
    Vorgehen:
    Es wird ein rechteckiges Fenster um den Bildmittelpunkt definiert. Alles 
    außerhalb dieses Fensters (die äußeren 15%, wenn keep_frac=0.85) wird maskiert.
    """
    h, w = mask.shape
    ch, cw = h // 2, w // 2  # Bildmittelpunkt
    
    # Berechnung der halben Fenstergröße basierend auf dem Anteil 'keep_frac'
    hh, ww = int(h * keep_frac / 2), int(w * keep_frac / 2)

    # Erstellung der ROI-Maske
    m2 = np.zeros_like(mask, dtype=bool)
    m2[ch - hh: ch + hh, cw - ww: cw + ww] = True
    
    # Logische UND-Verknüpfung: Pixel muss Brain-Maske UND im Zentrum sein
    return mask & m2

# ============================================================
# 3) K-Means Clustering (Segmentierung)
# ============================================================

def kmeans_fixed_init(X, init_centers, max_iter=40):
    """
    Führt den K-Means-Algorithmus zur Gewebeklassifizierung durch.
    
    Unterschied zu Standard-Implementierungen (z.B. sklearn):
    Dieser Algorithmus ist vollständig deterministisch. Er nutzt fest definierte
    Startzentren (init_centers) anstelle von zufälliger Initialisierung.
    Dies garantiert reproduzierbare Ergebnisse für medizinische Analysen.
    
    Funktionsweise (Lloyd-Algorithmus):
    1. Assignment: Jeder Voxel wird dem Clusterzentrum zugeordnet, zu dem er 
       die geringste euklidische Distanz hat.
    2. Update: Die Zentren werden neu berechnet als Mittelwert aller Voxel, 
       die ihnen zugeordnet wurden.
    Dieser Prozess wird wiederholt, bis sich die Zuordnungen nicht mehr ändern (Konvergenz).
    
    Parameter:
        X: Feature-Matrix (N Pixel x D Modalitäten).
        init_centers: Initiale Clusterzentren (K Klassen x D Modalitäten).
    """
    centers = init_centers.astype(np.float32).copy()
    K = centers.shape[0]  # Anzahl der Gewebeklassen (hier 3: CSF, GM, WM)

    labels = None
    for _ in range(max_iter):
        # 1. Berechnung der Distanzen aller Punkte zu allen Zentren
        #    X[:, None, :] shape (N, 1, D) - centers[None, :, :] shape (1, K, D)
        #    Ergebnis Broadcasting: (N, K, D) -> Norm über D -> (N, K)
        dist = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        
        # 2. Zuordnung zum nächsten Zentrum (Nearest Neighbor)
        new_labels = np.argmin(dist, axis=1)

        # 3. Konvergenzprüfung: Wenn sich Labels nicht ändern, Abbruch
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # 4. Update der Cluster-Zentren
        for k in range(K):
            pts = X[labels == k]
            if pts.shape[0] > 0:
                # Zentrum wandert zum Schwerpunkt der Punktwolke
                centers[k] = pts.mean(axis=0)
            # Falls Cluster leer ist, behalten wir das alte Zentrum bei (Stabilität)

    return labels, centers

def labels_to_segmap(labels, idx, shape):
    """
    Rekonstruiert das 2D-Bild aus den flachen Label-Vektoren.
    
    Die K-Means-Berechnung erfolgt nur auf den maskierten Pixeln (Vektor 'labels').
    Diese Funktion schreibt die Ergebnisse zurück an die korrekten x,y-Positionen
    im Bildarray. Labels werden um +1 verschoben (0=Hintergrund, 1=CSF, 2=GM, 3=WM).
    """
    seg = np.zeros(shape, dtype=np.uint8)
    # Zuweisung der Labels an die Positionen, die durch 'idx' (Masken-Indizes) definiert sind
    seg[idx[0], idx[1]] = (labels.astype(np.uint8) + 1)
    return seg

def create_rgb(seg):
    """
    Erstellt ein farbcodiertes Overlay für die visuelle Inspektion.
    
    Farbkodierung (Standard in vielen Neuroimaging-Tools):
    - Label 1 (CSF - Liquor): Blau
    - Label 2 (GM - Graue Substanz): Grün
    - Label 3 (WM - Weiße Substanz): Rot
    """
    rgb = np.zeros(seg.shape + (3,), dtype=np.float32)
    rgb[seg == 1, 2] = 1.0  # Blue channel
    rgb[seg == 2, 1] = 1.0  # Green channel
    rgb[seg == 3, 0] = 1.0  # Red channel
    return rgb

# ============================================================
# 4) Datenbasierte Initialisierung (Quantile)
# ============================================================

def compute_centers_from_quantiles(s_t1, s_fl, s_ir, mask,
                                   q_csf=0.15, q_gm=0.55, q_wm=0.90,
                                   band=10):
    """
    Berechnet robuste Startzentren für K-Means basierend auf der Statistik des T1-Bildes.
    
    Hintergrund:
    K-Means ist empfindlich gegenüber der Initialisierung. Zufällige Startpunkte können
    dazu führen, dass Gewebe vertauscht werden (z.B. Cluster 0 wird WM statt CSF).
    Da T1-Bilder anatomisch gut definiert sind (Dunkel=CSF, Mittel=GM, Hell=WM),
    nutzen wir Quantile der T1-Intensität zur Schätzung.
    
    Vorgehen:
    1. T1: Bestimmung der Intensitäten bei 15% (CSF), 55% (GM) und 90% (WM) der Verteilung.
    2. FLAIR/IR: Da die Kontraste hier anders sind, suchen wir Pixel, die im T1-Bild
       nahe an den oben berechneten Werten liegen (innerhalb eines 'Bandes'). 
       Von diesen Pixeln nehmen wir den Median im FLAIR- und IR-Bild.
       Dies stellt sicher, dass wir einen multispektralen Vektor erhalten, der physikalisch
       zusammenpasst.
    """
    # Extraktion der Intensitätswerte innerhalb der Gehirnmaske
    t = s_t1[mask].astype(np.float32)
    f = s_fl[mask].astype(np.float32)
    r = s_ir[mask].astype(np.float32)

    # Fallback für sehr kleine/leere Masken
    if t.size < 50:
        return (
            np.array([20,  80,  80], dtype=np.float32),
            np.array([110, 110, 110], dtype=np.float32),
            np.array([235, 170, 170], dtype=np.float32)
        )

    # 1. Bestimmung der T1-Stützstellen (Ankerpunkte)
    t_csf = float(np.quantile(t, q_csf))  # Repräsentativ für Liquor
    t_gm  = float(np.quantile(t, q_gm))   # Repräsentativ für Graue Substanz
    t_wm  = float(np.quantile(t, q_wm))   # Repräsentativ für Weiße Substanz

    # 2. Definition von Bändern um diese T1-Werte
    csf_band = t <= (t_csf + band)
    gm_band  = (t >= (t_gm - band)) & (t <= (t_gm + band))
    wm_band  = t >= (t_wm - band)

    # 3. Übertragung auf FLAIR und IR mittels Median
    f_med = float(np.median(f))
    r_med = float(np.median(r))

    def band_median(arr, band_mask, fallback):
        # Berechnet Median nur für Pixel im jeweiligen Band
        return float(np.median(arr[band_mask])) if np.any(band_mask) else float(fallback)

    # Zusammenstellen der 3D-Startvektoren [T1, FLAIR, IR]
    c_csf = np.array([t_csf, band_median(f, csf_band, f_med), band_median(r, csf_band, r_med)], dtype=np.float32)
    c_gm  = np.array([t_gm,  band_median(f, gm_band,  f_med), band_median(r, gm_band,  r_med)], dtype=np.float32)
    c_wm  = np.array([t_wm,  band_median(f, wm_band,  f_med), band_median(r, wm_band,  r_med)], dtype=np.float32)

    return c_csf, c_gm, c_wm

# ============================================================
# 5) Haupt-Pipeline pro Patient
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
    """
    Führt die vollständige Verarbeitungspipeline für einen Patienten durch.
    
    Schritte:
    1. Daten laden und Normalisieren (0-255).
    2. Slice-Extraktion (Standard: Mitte des Volumens).
    3. Maskierung (Skull Stripping + ROI).
    4. Feature-Erstellung: Kombination von T1, FLAIR und IR zu Vektoren.
    5. Gewichtung der Features: Da T1 und FLAIR oft kontrastreicher sind,
       können sie stärker gewichtet werden.
    6. Initialisierung und Clustering mittels K-Means.
    7. Rücktransformation und Aufbereitung der Ergebnisse.
    """

    print(f"\n=== Verarbeite Patient {patient_id} ===")

    # 1. Laden der Rohdaten
    t1 = nib.load(t1_path).get_fdata()
    fl = nib.load(flair_path).get_fdata()
    ir = nib.load(ir_path).get_fdata()

    # 2. Robuste Normalisierung auf 8-Bit
    t1_8 = robust_minmax_to_uint8(t1, p_low, p_high)
    fl_8 = robust_minmax_to_uint8(fl, p_low, p_high)
    ir_8 = robust_minmax_to_uint8(ir, p_low, p_high)

    # 3. Auswahl des Slices (z.B. Mitte des Gehirns)
    depth = t1_8.shape[2]
    if z_mode == "mid":
        z = depth // 2
    else:
        z = int(depth * float(z_mode))

    # Orientierungskorrektur der 2D-Slices
    s_t1 = orient_slice(t1_8, z)
    s_fl = orient_slice(fl_8, z)
    s_ir = orient_slice(ir_8, z)

    # 4. Maskenerstellung (Skull Stripping) auf Basis des T1-Bildes
    mask = get_brain_mask(s_t1, thr_rel=thr_rel)
    if use_center_constraint:
        mask = apply_center_constraint(mask, keep_frac=keep_frac)

    # Extraktion der Pixelkoordinaten innerhalb der Maske
    idx = np.where(mask)
    if idx[0].size == 0:
        raise RuntimeError("Fehler: Leere Maske. Bitte Schwellenwerte prüfen.")

    # 5. Feature-Matrix erstellen
    # Matrix X hat Dimension (Anzahl Pixel, 3 Modalitäten)
    X = np.stack([s_t1[idx], s_fl[idx], s_ir[idx]], axis=1).astype(np.float32)

    # Anwendung der Feature-Gewichtung
    # Dies verzerrt den Feature-Raum, sodass wichtige Modalitäten mehr Einfluss auf die Distanzberechnung haben
    w = np.array(weights, dtype=np.float32)
    Xw = X * w

    # 6. K-Means Initialisierung (Datenbasiert)
    c_csf, c_gm, c_wm = compute_centers_from_quantiles(
        s_t1, s_fl, s_ir, mask,
        q_csf=q_csf, q_gm=q_gm, q_wm=q_wm,
        band=band
    )

    # Skalierung der Startzentren mit den gleichen Gewichten wie die Daten
    init_centers = np.stack([c_csf * w, c_gm * w, c_wm * w], axis=0)

    # 7. Clustering durchführen
    labels, centers_w = kmeans_fixed_init(Xw, init_centers, max_iter=40)

    # 8. Rekonstruktion des Segmentierungsbildes
    seg = labels_to_segmap(labels, idx, s_t1.shape)

    # Rückrechnung der Zentren auf originale Skala für Interpretierbarkeit
    centers_unweighted = centers_w / (w + 1e-12)

    # Rückgabe aller relevanten Daten für Visualisierung und Debugging
    return {
        "patient_id": patient_id,
        "z": z,
        "t1": s_t1,
        "flair": s_fl,
        "ir": s_ir,
        "mask": mask,
        "seg": seg,
        "seg_rgb": create_rgb(seg),
        "init_centers_unweighted": np.stack([c_csf, c_gm, c_wm], axis=0),
        "final_centers_unweighted": centers_unweighted
    }

# ============================================================
# 6) Ausführung: Analyse für Patient 7 und 13 + Visualisierung
# ============================================================

# Parameterwahl basierend auf empirischen Tests für optimale Segmentierung
res7 = segment_patient_quantile_centers(
    "data/pat7_reg_T1.nii", "data/pat7_reg_FLAIR.nii", "data/pat7_reg_IR.nii", 
    patient_id=7, 
    thr_rel=0.10,          # 10% Schwellenwert für Maske
    use_center_constraint=True, 
    keep_frac=0.85,        # Entfernt äußere 15% des Bildes (Artefaktreduktion)
    q_csf=0.15, q_gm=0.55, q_wm=0.90, # Quantile zur Zentren-Schätzung
    band=10,
    weights=(1.2, 1.2, 0.8) # T1 und FLAIR sind informativer als IR, daher höher gewichtet
)

res13 = segment_patient_quantile_centers(
    "data/pat13_reg_T1.nii", "data/pat13_reg_FLAIR.nii", "data/pat13_reg_IR.nii", 
    patient_id=13, 
    thr_rel=0.10, 
    use_center_constraint=True, 
    keep_frac=0.85, 
    q_csf=0.15, q_gm=0.55, q_wm=0.90, 
    band=10, 
    weights=(1.2, 1.2, 0.8)
)

# 
# 

# Visualisierung der Ergebnisse
fig, axs = plt.subplots(2, 5, figsize=(22, 10))

for r, res in enumerate([res7, res13]):
    # Spalte 1: T1-Originalbild (Anatomische Referenz)
    axs[r, 0].imshow(res["t1"], cmap="gray", origin="lower")
    axs[r, 0].set_title(f"Pat{res['patient_id']} T1 (0..255), z={res['z']}")
    axs[r, 0].axis("off")

    # Spalte 2: Brain Mask (Qualitätskontrolle des Skull Strippings)
    axs[r, 1].imshow(res["t1"], cmap="gray", origin="lower")
    axs[r, 1].imshow(res["mask"], alpha=0.35, origin="lower", cmap='Reds') # Rot overlay
    axs[r, 1].set_title("Brain Mask (Center-Constraint)")
    axs[r, 1].axis("off")

    # Spalte 3: Segmentierung Overlay (Visueller Abgleich mit Anatomie)
    axs[r, 2].imshow(res["t1"], cmap="gray", origin="lower")
    axs[r, 2].imshow(res["seg_rgb"], alpha=0.55, origin="lower")
    axs[r, 2].set_title("Segmentierung (K-Means)")
    axs[r, 2].axis("off")

    # Spalte 4: Reine Labelmap (Diskrete Klassen)
    axs[r, 3].imshow(res["seg"], cmap="tab10", origin="lower")
    axs[r, 3].set_title("Klassen: 1=CSF, 2=GM, 3=WM")
    axs[r, 3].axis("off")

    # Spalte 5: Numerische Statistik der Clusterzentren
    axs[r, 4].axis("off")
    ic = res["init_centers_unweighted"]
    fc = res["final_centers_unweighted"]
    
    # Formatierung der Textausgabe für Cluster-Werte
    txt = (
        "Init-Zentren (T1, FL, IR)\n"
        f"CSF: {np.round(ic[0], 1)}\nGM : {np.round(ic[1], 1)}\nWM : {np.round(ic[2], 1)}\n\n"
        "Final-Zentren (Konvergenz)\n"
        f"CSF: {np.round(fc[0], 1)}\nGM : {np.round(fc[1], 1)}\nWM : {np.round(fc[2], 1)}"
    )
    axs[r, 4].text(0.0, 0.5, txt, fontsize=10, va="center", family='monospace')

plt.tight_layout()
plt.show()