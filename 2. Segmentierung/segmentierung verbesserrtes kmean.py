import numpy as np  
import nibabel as nib  
import matplotlib.pyplot as plt  
from scipy.ndimage import binary_fill_holes, label  

# ============================================================
# 1) Robust 0..255 Normierung (pro 3D-Volumen, deterministisch)
# ============================================================

def robust_minmax_to_uint8(vol, p_low=1, p_high=99):  # Funktion: robuste lineare Skalierung eines Volumens auf [0,255]
    """
    Robustes Min-Max-Scaling auf 0..255 (uint8) über ein ganzes 3D-Volumen.
    Clip an Perzentilen reduziert Ausreißer und macht es stabiler als min/max.
    """
    v = vol.astype(np.float32)  # Konvertiert Intensitäten in float32 für stabile Gleitkomma-Arithmetik
    lo = np.percentile(v, p_low)  # Minimum
    hi = np.percentile(v, p_high)  # Maximum
    if hi <= lo + 1e-6:  # Degenerationscheck: vermeidet Division durch ~0 (numerische Stabilität)
        return np.zeros_like(v, dtype=np.uint8)  # Rückgabe: komplett 0, falls keine Dynamik vorhanden ist

    v = np.clip(v, lo, hi)  # Clipping: begrenzt Werte auf [lo,hi] (Sättigung / Robustheit gegen Ausreißer)
    v = (v - lo) / (hi - lo)  # Lineare Normierung: affine Transformation auf [0,1]
    return (v * 255.0).astype(np.uint8)  # Skalierung [0,1]→[0,255] und Quantisierung auf uint8 (Diskretisierung)

def orient_slice(vol_uint8, z):  # Funktion: Slice entnehmen und Orientierung konsistent anpassen
    """
    Gleiche Orientierung wie in deinem bisherigen Code:
    transpose + flip lr + flip ud.
    """
    s = np.transpose(vol_uint8[:, :, z])  # Transponiert Achsen (Matrix-Transpose) zur Achsenkorrektur
    s = np.fliplr(s)  # Links-Rechts-Spiegelung (Index-Reversal in x-Richtung)
    s = np.flipud(s)  # Oben-Unten-Spiegelung (Index-Reversal in y-Richtung)
    return s  # Gibt den orientierten 2D-Slice zurück

# ============================================================
# 2) Brain Mask + optionaler Center-Constraint
# ============================================================

def get_brain_mask(slice_uint8, thr_rel=0.10):  
    
    """
     Brain-Maske auf dem 0..255 Slice.
    thr_rel=0.10 bedeutet: Schwelle = 10% von max (max=255 -> 25.5).
    """
    thresh = thr_rel * float(np.max(slice_uint8))  # Relativer Schwellwert: T = α·max (Skalierung nach Maximalwert)
    m = slice_uint8 > thresh  # Binärsegmentierung: Indikatorfunktion I(x)>T (Schwellwertklassifikation)
    m = binary_fill_holes(m)  # Lochfüllung: füllt innenliegende 0-Regionen in 1-Komponenten (Morphologische Rekonstruktion)

    lbl, num = label(m)  # Connected-Components: vergibt Labels je zusammenhängender 1-Region (Graph/4- oder 8-Nachbarschaft)
    if num == 0:  # Sonderfall: keine Komponente gefunden
        return np.zeros_like(slice_uint8, dtype=bool)  # Rückgabe: leere Maske (alles False)

    sizes = np.bincount(lbl.ravel())  # Histogramm der Labelhäufigkeiten (Zählen der Pixel pro Komponente)
    sizes[0] = 0  # Label 0 = Hintergrund, wird ignoriert (damit nicht als größte Komponente gewählt)
    return lbl == sizes.argmax()  # Größte Komponente: argmax wählt größte Fläche (Maximum-Likelihood unter Flächenprior)

def apply_center_constraint(mask, keep_frac=0.85):  
    """
    Schneidet die Maske auf ein zentriertes Rechteck.
    Das entfernt häufig Nase/Augen/Frontartefakte deterministisch.
    keep_frac=0.85 behält 85% von Breite/Höhe um die Bildmitte.
    """
    h, w = mask.shape  # Bildhöhe und -breite (Geometrie des 2D-Gitters)
    ch, cw = h // 2, w // 2  # Mittelpunktkoordinate (integer floor; diskretes Koordinatensystem)
    hh, ww = int(h * keep_frac / 2), int(w * keep_frac / 2)  # Halbe Fenstergröße: (keep_frac·Dimension)/2

    m2 = np.zeros_like(mask, dtype=bool)  # Initialisiert ROI-Maske (alles False)
    m2[ch - hh: ch + hh, cw - ww: cw + ww] = True  # Setzt zentralen Rechteckbereich True (Ausschnitt/Windowing)
    return mask & m2  # Logisches AND: Schnittmenge der Masken (M ∩ ROI)

# ============================================================
# 3) K-Means (deterministisch) mit festen/datenbasierten Startzentren
# ============================================================

def kmeans_fixed_init(X, init_centers, max_iter=40):  # K-Means: Minimiert Summe quadratischer Abstände zu Zentren
    """
    K-Means auf X (N, D) mit festen Zentren (K, D).
    Deterministisch, kein Zufall. Wenn ein Cluster leer wird, bleibt
    sein Zentrum unverändert (stabiler als zufälliges reseeding).
    """
    centers = init_centers.astype(np.float32).copy()  # Startzentren (K,D) als float32, Kopie für Updates
    K = centers.shape[0]  # Anzahl Cluster K (hier typischerweise 3 für CSF/GM/WM)

    labels = None  # Platzhalter für Zuordnung jedes Punkts zu einem Cluster
    for _ in range(max_iter):  # Iterationen: Lloyd-Algorithmus (wechselnd Assignment/Update)
        dist = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)  # Euklidische Distanzmatrix ||x_i-c_k|| (L2-Norm)
        new_labels = np.argmin(dist, axis=1)  # Assignment: wähle k mit minimaler Distanz (Nearest-Center-Voronoi)

        if labels is not None and np.array_equal(new_labels, labels):  # Konvergenz: keine Labeländerung → Fixpunkt
            break  # Abbruch, da weiterer Durchlauf keine Änderung bringt
        labels = new_labels  # Aktualisiert aktuelle Clusterzuordnung

        for k in range(K):  # Update-Schritt: Zentren neu berechnen
            pts = X[labels == k]  # Alle Punkte im Cluster k (Indexfilter)
            if pts.shape[0] > 0:  # Nur wenn Cluster nicht leer ist
                centers[k] = pts.mean(axis=0)  # Mittelwert = Minimierer der SSE im Cluster (Least Squares / L2-Optimalität)
            # else: Zentrum bleibt unverändert  # Leeres Cluster: kein Update, damit deterministisch und stabil

    return labels, centers  # Rückgabe: Labels pro Punkt und finale Zentren

def labels_to_segmap(labels, idx, shape):  # Baut aus Punktlabels wieder ein 2D-Labelbild
    seg = np.zeros(shape, dtype=np.uint8)  # Initialisiert Labelmap: 0 = Hintergrund
    seg[idx[0], idx[1]] = (labels.astype(np.uint8) + 1)  # Schreibt 1..3 in die Maskenpixel (Labelverschiebung)
    return seg  # Gibt 2D-Segmentierungsmap zurück

def create_rgb(seg):  # Umwandlung Labelmap → RGB-Overlay (Farbcodierung)
    rgb = np.zeros(seg.shape + (3,), dtype=np.float32)  # RGB-Array (H,W,3) initial 0 (schwarz)
    rgb[seg == 1, 2] = 1.0  # CSF: blau (B-Kanal auf 1)
    rgb[seg == 2, 1] = 1.0  # GM: grün (G-Kanal auf 1)
    rgb[seg == 3, 0] = 1.0  # WM: rot (R-Kanal auf 1)
    return rgb  # Gibt farbiges Overlay zurück

# ============================================================
# 4) Neue Startzentren: deterministisch aus Quantilen (pro Patient passend)
# ============================================================

def compute_centers_from_quantiles(s_t1, s_fl, s_ir, mask,
                                  q_csf=0.15, q_gm=0.55, q_wm=0.90,
                                  band=10):  # Startzentren über Quantile (robuste Lageparameter)
    """
    Deterministische Startzentren aus Quantilen (T1) im Gehirnmaskenbereich.

    Idee:
      - CSF: niedriges T1-Quantil
      - GM : mittleres T1-Quantil
      - WM : hohes T1-Quantil
    FLAIR/IR-Zentrum wird als Median in einem Intensitätsband um das jeweilige
    T1-Quantil bestimmt, damit Zentren wirklich 3D (T1/FLAIR/IR) sind.
    """
    t = s_t1[mask].astype(np.float32)  # T1-Werte im Gehirn: Stichprobe aus Maskenbereich
    f = s_fl[mask].astype(np.float32)  # FLAIR-Werte im Gehirn: gleiche Pixelpositionen
    r = s_ir[mask].astype(np.float32)  # IR-Werte im Gehirn: gleiche Pixelpositionen

    if t.size < 50:  # Sicherheitsfallback: zu wenig Punkte für robuste Statistik
        # sehr kleine Maske -> fallback: fixe Zentren
        return (
            np.array([20,  80,  80], dtype=np.float32),  # CSF: niedrige Intensitäten
            np.array([110, 110, 110], dtype=np.float32),  # GM: mittlere Intensitäten
            np.array([235, 170, 170], dtype=np.float32)  # WM: hohe Intensitäten
        )

    t_csf = float(np.quantile(t, q_csf))  # Quantil als robuster „niedriger“ Lagewert (Ordnungstatistik)
    t_gm  = float(np.quantile(t, q_gm))  # Quantil als „mittlerer“ Lagewert (robust gegen Ausreißer)
    t_wm  = float(np.quantile(t, q_wm))  # Quantil als „hoher“ Lagewert (Ordnungstatistik)

    # Bänder definieren (auf T1)
    csf_band = t <= (t_csf + band)  # Bereich „nahe CSF“: T1 unterhalb (t_csf + band)
    gm_band  = (t >= (t_gm - band)) & (t <= (t_gm + band))  # Bereich „nahe GM“: symmetrisches Band um t_gm
    wm_band  = t >= (t_wm - band)  # Bereich „nahe WM“: T1 oberhalb (t_wm - band)

    f_med = float(np.median(f))  # Median FLAIR als robuster Zentralwert (L1-Optimalität)
    r_med = float(np.median(r))  # Median IR als robuster Zentralwert (L1-Optimalität)

    def band_median(arr, band_mask, fallback):  # Hilfsfunktion: Median in Band, sonst Fallback
        return float(np.median(arr[band_mask])) if np.any(band_mask) else float(fallback)  # Median = robust, Ausreißer-resistent

    c_csf = np.array([t_csf, band_median(f, csf_band, f_med), band_median(r, csf_band, r_med)], dtype=np.float32)  # CSF-Zentrum (T1,FL,IR)
    c_gm  = np.array([t_gm,  band_median(f, gm_band,  f_med), band_median(r, gm_band,  r_med)], dtype=np.float32)  # GM-Zentrum
    c_wm  = np.array([t_wm,  band_median(f, wm_band,  f_med), band_median(r, wm_band,  r_med)], dtype=np.float32)  # WM-Zentrum

    return c_csf, c_gm, c_wm  # Rückgabe der drei Startzentren (ungewichtet)

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
                                    weights=(1.2, 1.2, 0.8)):  # Hauptpipeline: Laden → Normieren → Maske → Features → K-Means
    """
    Ablauf:
      - 3D Volumen je Modalität robust auf 0..255 normieren
      - Slice wählen
      - Brain Mask auf T1 anwenden (+ optional Center-Constraint)
      - Features (T1, FLAIR, IR) aus Maske extrahieren
      - K-Means deterministisch mit datenbasierten Startzentren (Quantile)
    """

    print(f"\n=== Patient {patient_id} ===")  # Logging: zeigt, welcher Patient verarbeitet wird

    t1 = nib.load(t1_path).get_fdata()  # Lädt T1-Volumen als float (Voxelintensitäten)
    fl = nib.load(flair_path).get_fdata()  # Lädt FLAIR-Volumen als float
    ir = nib.load(ir_path).get_fdata()  # Lädt IR-Volumen als float

    # 0..255 Normierung über das gesamte Volumen (robust)
    t1_8 = robust_minmax_to_uint8(t1, p_low, p_high)  # T1: robuste affine Skalierung und Quantisierung auf uint8
    fl_8 = robust_minmax_to_uint8(fl, p_low, p_high)  # FLAIR: gleiche Skalierungsmethode
    ir_8 = robust_minmax_to_uint8(ir, p_low, p_high)  # IR: gleiche Skalierungsmethode

    depth = t1_8.shape[2]  # Anzahl Slices in z-Richtung (3. Dimension)
    if z_mode == "mid":  # Fall: mittlerer Slice
        z = depth // 2  # Index des mittleren Slices (Integer-Division)
    else:
        z = int(depth * float(z_mode))  # Alternative: relativer Sliceanteil (z_mode in [0,1])

    s_t1 = orient_slice(t1_8, z)  # Entnimmt T1-Slice z und korrigiert Orientierung
    s_fl = orient_slice(fl_8, z)  # Entnimmt FLAIR-Slice z und korrigiert Orientierung
    s_ir = orient_slice(ir_8, z)  # Entnimmt IR-Slice z und korrigiert Orientierung

    # Maske
    mask = get_brain_mask(s_t1, thr_rel=thr_rel)  # Skull-Stripping grob über Schwellwert + größte Komponente
    if use_center_constraint:  # Option: zusätzliche geometrische Einschränkung
        mask = apply_center_constraint(mask, keep_frac=keep_frac)  # ROI-Schnitt: entfernt häufig frontale Nicht-Gehirn-Anteile

    idx = np.where(mask)  # Indexliste aller True-Pixel (Koordinaten der Maskenpunkte)
    if idx[0].size == 0:  # Sonderfall: keine Pixel in Maske
        raise RuntimeError("Maske ist leer. thr_rel reduzieren oder keep_frac erhöhen.")  # Harte Fehlerausgabe zur Diagnose

    # Features
    X = np.stack([s_t1[idx], s_fl[idx], s_ir[idx]], axis=1).astype(np.float32)  # Featurevektor je Pixel: (T1,FLAIR,IR)

    w = np.array(weights, dtype=np.float32)  # Gewichtungsvektor (Feature-Skalierung → anisotrope Distanzmetrik)
    Xw = X * w  # Gewichtet: entspricht einer Diagonalmetrik in L2 (||W(x-c)||)

    # Startzentren deterministisch aus Quantilen
    c_csf, c_gm, c_wm = compute_centers_from_quantiles(  # Schätzt initiale Zentren robust aus der Datenverteilung
        s_t1, s_fl, s_ir, mask,  # Übergibt die Modalitäten und die Gehirnmaske
        q_csf=q_csf, q_gm=q_gm, q_wm=q_wm,  # Quantile definieren low/mid/high
        band=band  # Bandbreite um Quantile zur Medianbestimmung in FLAIR/IR
    )

    init_centers = np.stack([c_csf * w, c_gm * w, c_wm * w], axis=0)  # Zentren in den gewichteten Raum abbilden (K,D)

    labels, centers_w = kmeans_fixed_init(Xw, init_centers, max_iter=40)  # Lloyd-K-Means: Assignment/Update bis Konvergenz

    seg = labels_to_segmap(labels, idx, s_t1.shape)  # Schreibt Punktlabels zurück in 2D-Bildkoordinaten (Segmentkarte)

    # Zur Interpretation ungewichtet zurück
    centers_unweighted = centers_w / (w + 1e-12)  # Rücktransformation aus Gewichtung (inverse Skalierung; eps gegen /0)

    return {  # Sammeln aller Ergebnisse in einem Dictionary (strukturiertes Ergebnisobjekt)
        "patient_id": patient_id,  # Patient-ID (für Anzeige/Debug)
        "z": z,  # verwendeter Slice-Index
        "t1": s_t1,  # T1-Slice (0..255)
        "flair": s_fl,  # FLAIR-Slice (0..255)
        "ir": s_ir,  # IR-Slice (0..255)
        "mask": mask,  # finale Gehirnmaske
        "seg": seg,  # Labelmap (0=BG, 1=CSF, 2=GM, 3=WM)
        "seg_rgb": create_rgb(seg),  # RGB-Overlay für Visualisierung
        "init_centers_unweighted": np.stack([c_csf, c_gm, c_wm], axis=0),  # Startzentren im ungewichteten Raum (Interpretation)
        "final_centers_unweighted": centers_unweighted  # Endzentren im ungewichteten Raum (Interpretation)
    }

# ============================================================
# 6) Demo: Patient 7 und 13 + Visualisierung
# ============================================================

res7 = segment_patient_quantile_centers(
    "pat7_reg_T1.nii.gz", "pat7_reg_FLAIR.nii.gz", "pat7_reg_IR.nii.gz", 
    patient_id=7, 
    thr_rel=0.10,  # Maskenschwellwert relativ zu max (Schwellwertsegmentierung)
    use_center_constraint=True, 
    keep_frac=0.85,  # Anteil des Bildes, der um die Mitte behalten wird
    q_csf=0.15, q_gm=0.55, q_wm=0.90,  # Quantile für CSF/GM/WM (robuste Lageparameter)
    band=10,  # Bandbreite um Quantile zur Medianbestimmung in FLAIR/IR
    weights=(1.2, 1.2, 0.8)  # Featuregewichtung (Distanzmetrik: T1/FLAIR stärker als IR)
)

res13 = segment_patient_quantile_centers(  # Führt die Pipeline für Patient 13 aus
    "pat13_reg_T1.nii.gz", "pat13_reg_FLAIR.nii.gz", "pat13_reg_IR.nii.gz",  # Pfade zu den 3 Modalitäten
    patient_id=13,  # ID für Logging/Anzeige
    thr_rel=0.10,  # Maskenschwellwert relativ zu max
    use_center_constraint=True,  # Aktiviert Center-ROI-Schnitt
    keep_frac=0.85,  # Anteil des Bildes, der um die Mitte behalten wird
    q_csf=0.15, q_gm=0.55, q_wm=0.90,  # Quantile für CSF/GM/WM
    band=10,  # Bandbreite
    weights=(1.2, 1.2, 0.8)  # Featuregewichtung
)

fig, axs = plt.subplots(2, 5, figsize=(22, 10))  # Erzeugt 2x5 Subplot-Gitter für Vergleich Pat7/Pat13
for r, res in enumerate([res7, res13]):  # Iteriert über beide Ergebnisse (Zeilen im Plot)
    axs[r, 0].imshow(res["t1"], cmap="gray", origin="lower")  # Zeigt T1-Slice als Graustufenbild
    axs[r, 0].set_title(f"Pat{res['patient_id']} T1 (0..255), z={res['z']}")  # Titel mit Patient und Sliceindex
    axs[r, 0].axis("off")  # Achsen ausblenden (nur Bild)

    axs[r, 1].imshow(res["t1"], cmap="gray", origin="lower")  # Hintergrund: T1
    axs[r, 1].imshow(res["mask"], alpha=0.35, origin="lower")  # Overlay: Maske mit Transparenz (Alpha-Blending)
    axs[r, 1].set_title("Brain Mask (mit Center-Constraint)")  # Titel
    axs[r, 1].axis("off")  # Achsen aus

    axs[r, 2].imshow(res["t1"], cmap="gray", origin="lower")  # Hintergrund: T1
    axs[r, 2].imshow(res["seg_rgb"], alpha=0.55, origin="lower")  # Overlay: farbige Segmentierung (Alpha-Blending)
    axs[r, 2].set_title("Segmentierung (K-Means deterministisch)")  # Titel
    axs[r, 2].axis("off")  # Achsen aus

    axs[r, 3].imshow(res["seg"], cmap="tab10", origin="lower")  # Zeigt Labelmap mit diskreter Colormap
    axs[r, 3].set_title("Labels (1=CSF,2=GM,3=WM)")  # Titel erklärt Labelkodierung
    axs[r, 3].axis("off")  # Achsen aus

    axs[r, 4].axis("off")  # Fünftes Feld: nur Textinfos, keine Achsen
    ic = res["init_centers_unweighted"]  # Startzentren (ungewichtet) für Interpretation
    fc = res["final_centers_unweighted"]  # Endzentren (ungewichtet) für Interpretation
    txt = (  # Baut Textblock für die Anzeige (Debug/Interpretation)
        "Init-Zentren (T1,FL,IR)\n"
        f"CSF: {ic[0]}\nGM : {ic[1]}\nWM : {ic[2]}\n\n"
        "Final-Zentren\n"
        f"CSF: {fc[0]}\nGM : {fc[1]}\nWM : {fc[2]}"
    )
    axs[r, 4].text(0.0, 0.5, txt, fontsize=10, va="center")  # Schreibt Text in das Feld (Koordinaten im Achsensystem)

plt.tight_layout()  
plt.show()  
