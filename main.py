#Obaid
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
fig, axs = plt.subplots(1, 4, figsize=(20,5))

# T1 Original
axs[0].imshow(T1_slice, cmap="gray", origin="lower")
axs[0].set_title("T1 Original")
axs[0].axis("off")

# FLAIR Original
axs[1].imshow(FLAIR_slice, cmap="gray", origin="lower")
axs[1].set_title("FLAIR Original")
axs[1].axis("off")

# Brain-Maske Overlay
axs[2].imshow(T1_slice, cmap="gray", origin="lower")
axs[2].imshow(brain_mask_stripped, cmap="Reds", alpha=0.3)
axs[2].set_title("Skull-stripped Brain Mask")
axs[2].axis("off")

# Maskiertes Gehirn
axs[3].imshow(T1_brain_stripped, cmap="gray", origin="lower")
axs[3].set_title("T1 Skull-stripped Brain")
axs[3].axis("off")

plt.show()

print("\nT1 und FLAIR erfolgreich verarbeitet. Skull-stripped Brain erstellt.")
# Verena 
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, label

# -------------------------------------------------------
# 1) Hilfsfunktionen
# -------------------------------------------------------

def load_volume_corrected(path, name):
    """
    Lädt ein 3D-MRI-Volumen und korrigiert die Orientierung
    so, dass die Slices ähnlich aussehen wie bei 2D-Code.
    """
    print(f"\n--- Lade Volumen {name} ---")
    img = nib.load(path)
    data = np.asarray(img.dataobj).astype(float)   # (X, Y, Z)
    print(f"{name} Volumen-Form (roh):", data.shape)

    # wie beim 2D-Slice: transpose + Flip X/Y
    # 2D hattest du: slice = np.transpose(slice); flipud; fliplr
    # Das entspricht hier: Achsen 0 und 1 vertauschen, dann flip 0 und 1
    vol = np.transpose(data, (1, 0, 2))  # (Y, X, Z) -> (X', Y', Z)
    vol = np.flip(vol, axis=0)           # flipud
    vol = np.flip(vol, axis=1)           # fliplr

    # print(f"{name} Volumen-Form (korrigiert):", vol.shape)
    return vol


def get_brain_mask_slice(slice_img, threshold=0.05):
    """
    Dein 2D-Brain-Mask-Ansatz für EINEN Slice.
    """
    # 1. relativer Schwellenwert
    thresh_val = threshold * np.max(slice_img)
    mask = slice_img > thresh_val

    # 2. Löcher füllen
    mask_filled = binary_fill_holes(mask)

    # 3. größte zusammenhängende Komponente behalten
    labels2d, num = label(mask_filled)
    if num == 0:
        return np.zeros_like(slice_img, dtype=bool)
    sizes = np.bincount(labels2d.ravel())
    sizes[0] = 0  # Hintergrund ignorieren
    largest_label = sizes.argmax()
    brain_mask_clean = labels2d == largest_label
    return brain_mask_clean


# -------------------------------------------------------
# 2) Volumina laden (T1, FLAIR, R1)
# -------------------------------------------------------

T1_vol    = load_volume_corrected("data/pat13_reg_T1.nii",    "T1")
FLAIR_vol = load_volume_corrected("data/pat13_reg_FLAIR.nii", "FLAIR")
IR_vol    = load_volume_corrected("data/pat13_reg_IR.nii",    "IR")  # für später wichtig; irrelevant für Brain Mask 

nx, ny, nz = T1_vol.shape #nz = Anzahl der slices
print(f"Volumen-Abmessungen: nx={nx}, ny={ny}, nz={nz}")

# optional: Parameter aus deinem 2D-Code
brain_threshold     = 0.05
t1_upper_threshold  = 1200
flair_lower_threshold = 50

# -------------------------------------------------------
# 3) 3D-Brain-Maske aus allen Slices aufbauen
# -------------------------------------------------------

brain_mask_3d = np.zeros_like(T1_vol, dtype=bool) #leere Maske

for z in range(nz): #es wird jeder slice von 1 bis nz mit der Brain mask 'gebrainmaskt'
    T1_slice    = T1_vol[:, :, z]
    FLAIR_slice = FLAIR_vol[:, :, z]

    # grobe Brain-Masken pro Slice
    T1_mask    = get_brain_mask_slice(T1_slice, threshold=brain_threshold)
    FLAIR_mask = get_brain_mask_slice(FLAIR_slice, threshold=brain_threshold)

    brain_mask_combined = T1_mask | FLAIR_mask

    # intensity-basiertes Skull-Stripping 
    brain_mask_filtered = brain_mask_combined & (T1_slice < t1_upper_threshold)
    brain_mask_filtered = brain_mask_filtered & (FLAIR_slice > flair_lower_threshold)
    brain_mask_stripped = binary_fill_holes(brain_mask_filtered)

    brain_mask_3d[:, :, z] = brain_mask_stripped

#print("3D-Brain-Maske erstellt.")

# -------------------------------------------------------
# 4) 3D-K-Means-Cluster: CSF / GM / WM
# -------------------------------------------------------

brain_idx = np.where(brain_mask_3d) # Array, mit allen koordninaten (mit brain mask) alle slices
if brain_idx[0].size == 0:
    raise RuntimeError("3D-Brain-Maske ist leer – überprüfe Thresholds und Daten.")  # Abbruch des programms falls kein gehirn erkannt wird 

feat_T1    = T1_vol[brain_idx] #koordinaten von allen slices aus T1
feat_FLAIR = FLAIR_vol[brain_idx]#koordinaten von allen slices aus Flair
feat_IR    = IR_vol[brain_idx]#koordinaten von allen slices aus IR

features = np.stack([feat_T1, feat_FLAIR, feat_IR], axis=1).astype(float) # kombiniert alle 3 Feature Vektoren, zu einer matrix -> und formt in float um 

# Normierung pro Kanal
eps = 1e-6 # damit sigma nicht null werden kann 
features_norm = features.copy() #kopiert unsere Matrix 
for k in range(3): # k = T1, IR und Flair 
    mu    = np.mean(features_norm[:, k]) # mittelwert jeweils von allen werten von T1, Flair, IR 
    sigma = np.std(features_norm[:, k]) + eps #Standardabweichung für Normierung und Standardisierung, 
    features_norm[:, k] = (features_norm[:, k] - mu) / sigma #Normeirte Matrix, alle werte gleichwertig 

# --- K-Means mit K=3 ---
K = 3
max_iter = 20 #zahl nicht so wichtig 

rng = np.random.default_rng(0) # startzentrum weil k-mean das immer braucht, start position ist immer gleich bei llen slices 
rand_idx = rng.choice(features_norm.shape[0], size=K, replace=False) # zufallsgenerator: wähle K zufällige Voxel aus, dass dieselben mehrfach gewählt werden  
centers = features_norm[rand_idx, :]# diese werte werden als Clusterzentrum verwendet 

for it in range(max_iter):
    # Zuordnung zu nächsten Zentren
    distanz = np.linalg.norm(features_norm[:, None, :] - centers[None, :, :], axis=2) # Distanz zwischen voxel und zentrum
    labels = np.argmin(distanz, axis=1) #kleinste distanz 

    # Zentren neu berechnen
    new_centers = np.zeros_like(centers) # leere Mappe 
    for k in range(K):
        mask_k = labels == k # neues zentrum weil das die kleinste distanz hat also der beste wert 
        if np.any(mask_k):
            new_centers[k, :] = np.mean(features_norm[mask_k], axis=0) # neues zentrum; warum mittelwert ?
        else:
            new_centers[k, :] = features_norm[rng.integers(0, features_norm.shape[0])] #wenns kein neues zentrum gibt dann wird ein zufälliges aus der matrix gewählt 

    shift = np.linalg.norm(new_centers - centers)#berechnet abstand von allen zentren  
    centers = new_centers # überschreiben
    if shift < 1e-3: #wenn zentrum sich nciht mehr viel unterscheiden, dann haben wir schon ein sehr guten zentrum -> abbruch 
        break

print(f"K-Means nach {it+1} Iterationen konvergiert.")

# Cluster nach mittlerer T1-Intensität sortieren
cluster_T1_means = [] # für jeden cluster berechnet der code die mittlere t1 intensität und speichert sie zusmmen mit der cluster nummer in eine liste
for k in range(K):
    mean_T1_k = np.mean(feat_T1[labels == k]) if np.any(labels == k) else np.inf
    cluster_T1_means.append((k, mean_T1_k))

cluster_T1_means.sort(key=lambda x: x[1]) #sortiert von dunkel nach hell 
csf_label = cluster_T1_means[0][0]
gm_label  = cluster_T1_means[1][0]
wm_label  = cluster_T1_means[2][0]

print("Cluster-Zuordnung (nach T1-Mittelwert):")
print(f"  CSF (blau) = Cluster {csf_label}")
print(f"  GM  (grün) = Cluster {gm_label}")
print(f"  WM  (rot)  = Cluster {wm_label}")

# -------------------------------------------------------
# 5) 3D-Masken aufbauen und einfache Optimierung
# -------------------------------------------------------

csf_mask_3d = np.zeros_like(T1_vol, dtype=bool) #leere maken anlegen 
gm_mask_3d  = np.zeros_like(T1_vol, dtype=bool)
wm_mask_3d  = np.zeros_like(T1_vol, dtype=bool)

csf_mask_3d[brain_idx] = (labels == csf_label) # werte sortieren nach CSf GM und Wm und 3d speichern 
gm_mask_3d[brain_idx]  = (labels == gm_label)
wm_mask_3d[brain_idx]  = (labels == wm_label)

# alles außerhalb des Gehirns sicher ausschließen
csf_mask_3d &= brain_mask_3d
gm_mask_3d  &= brain_mask_3d
wm_mask_3d  &= brain_mask_3d

# Löcher im CSF schließen (3D)
csf_mask_3d = binary_fill_holes(csf_mask_3d)

# GM/WM nicht innerhalb des CSF zulassen
gm_mask_3d &= ~csf_mask_3d
wm_mask_3d &= ~csf_mask_3d

print("3D-Segmentierung in CSF / GM / WM abgeschlossen.")

# -------------------------------------------------------
# 6) Visualisierung: ein paar Slices anzeigen
# -------------------------------------------------------

def show_slice_with_seg(z):
    T1_slice = T1_vol[:, :, z]
    csf = csf_mask_3d[:, :, z]
    gm  = gm_mask_3d[:, :, z]
    wm  = wm_mask_3d[:, :, z]

    seg_rgb = np.zeros(T1_slice.shape + (3,), dtype=float)
    seg_rgb[csf, 2] = 1.0  # CSF -> blau
    seg_rgb[gm, 1]  = 1.0  # GM  -> grün
    seg_rgb[wm, 0]  = 1.0  # WM  -> rot

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(T1_slice, cmap="gray", origin="lower")
    axs[0].set_title(f"T1 Slice {z}")
    axs[0].axis("off")

    axs[1].imshow(T1_slice, cmap="gray", origin="lower")
    axs[1].imshow(seg_rgb, alpha=0.6, origin="lower")
    axs[1].set_title(f"Segmentierung Slice {z}")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()

# Beispiel: drei Slices anschauen (oben / Mitte / unten)
show_slice_with_seg(5)
show_slice_with_seg(nz // 2)
show_slice_with_seg(nz - 5)

# -------------------------------------------------------
# 7) ALLE SLICES IN EINEM GROßEN GRID PLOTTEN
# -------------------------------------------------------

def show_all_slices(T1_vol, csf_mask, gm_mask, wm_mask, cols=8):
    """
    Zeigt alle Slices als Miniaturbilder in einem einzigen großen Plot.
    cols = Anzahl Bilder pro Zeile
    """
    nz = T1_vol.shape[2]
    rows = int(np.ceil(nz / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))

    for z in range(nz):
        r = z // cols
        c = z % cols

        ax = axes[r, c] if rows > 1 else axes[c]

        # Slice-Daten
        T1_slice = T1_vol[:, :, z]
        csf = csf_mask[:, :, z]
        gm  = gm_mask[:, :, z]
        wm  = wm_mask[:, :, z]

        # RGB bauen
        seg_rgb = np.zeros(T1_slice.shape + (3,), dtype=float)
        seg_rgb[csf, 2] = 1.0  # blau
        seg_rgb[gm, 1]  = 1.0  # grün
        seg_rgb[wm, 0]  = 1.0  # rot

        # Plot
        ax.imshow(T1_slice, cmap="gray", origin="lower")
        ax.imshow(seg_rgb, alpha=0.5, origin="lower")
        ax.set_title(f"Slice {z}")
        ax.axis("off")

    # Falls mehr Felder als Slices, leere Felder ausblenden
    for r in range(rows):
        for c in range(cols):
            idx = r*cols + c
            if idx >= nz:
                axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()


# ---- AUFRUF ----
show_all_slices(T1_vol, csf_mask_3d, gm_mask_3d, wm_mask_3d, cols=8)