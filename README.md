import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, measure, morphology, feature, segmentation, util, exposure
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree, shortest_path, dijkstra
from scipy import ndimage
import pandas as pd
from pathlib import Path

# ======================== CONFIG ========================
IMAGE_PATH = r"H:\OneDrive - University of Nebraska-Lincoln\1.Collaboration\Ock\Image for Analysis\AuNPsSEM_onlyChain02.png"

# Image and segmentation
INVERT_FOR_WATERSHED = False    # particles bright on dark background -> keep False
GAUSS_SIGMA = 1.5               # blur to denoise before thresholding
USE_ADAPTHIST = False           # set True if illumination is uneven

# Markers for watershed
FOOTPRINT_PX = None             # None = auto estimate from regions; else set int like 7 or 11

# Graph connectivity
INTERACTION_DISTANCE_PIXELS = 40  # key knob that controls edges between particles

# Physical scaling (optional)
PIXEL_SIZE_NM = None            # set to a number to report nm

# Percolation
PERCOLATION_AXIS = 'left-right' # 'left-right' or 'top-bottom'
EDGE_MARGIN_PX = 5              # how close to border to count as touching

# Output saving (optional)
SAVE_PREFIX = None              # e.g., r"C:\temp\AuNPs_perc" to save PNGs and CSV
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)

# ===================== HELPERS ==========================
def robust_read_gray(path):
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = util.img_as_float(img)
    return img

def otsu_binary(im, invert=False):
    thr = filters.threshold_otsu(im)
    b = im > thr
    if invert:
        b = ~b
    b = morphology.remove_small_objects(b, min_size=16)
    b = morphology.remove_small_holes(b, area_threshold=16)
    return b

def auto_footprint_from_regions(label_img, default_fp=8):
    props = measure.regionprops(label_img)
    if not props:
        return default_fp
    equiv_r = [np.sqrt(p.area/np.pi) for p in props]
    r = np.median(equiv_r)
    fp = int(max(5, min(25, round(r*1.5))))
    return fp

def detect_markers(distance, mask, footprint_px):
    coords = feature.peak_local_max(distance,
                                    footprint=np.ones((footprint_px, footprint_px), dtype=bool),
                                    labels=mask)
    local_maxi = np.zeros(distance.shape, dtype=bool)
    if coords.size > 0:
        local_maxi[coords[:, 0], coords[:, 1]] = True
    markers = measure.label(local_maxi)
    return markers

def region_centroids_areas(lbl):
    props = measure.regionprops(lbl)
    cents = np.array([p.centroid for p in props]) if props else np.empty((0, 2))
    areas = np.array([p.area for p in props]) if props else np.empty((0,))
    return cents, areas, props

def build_adjacency(coords, thresh_px):
    n = len(coords)
    if n <= 1:
        return csr_matrix((n, n)), np.zeros((n, n))
    D = squareform(pdist(coords, 'euclidean'))
    A = (D < thresh_px) & (D > 0)
    return csr_matrix(A.astype(int)), D

def mst_tortuosity(coords, A, D):
    n = len(coords)
    if n <= 1:
        return 1.0
    W = np.where(A.toarray() > 0, D, 0.0)
    W_sparse = csr_matrix(W)
    mst = minimum_spanning_tree(W_sparse)
    sp_dists, _ = shortest_path(mst, directed=False, return_predecessors=True)
    i, j = np.unravel_index(np.nanargmax(sp_dists), sp_dists.shape)
    contour_len = sp_dists[i, j]
    end_to_end = np.linalg.norm(coords[i] - coords[j])
    if end_to_end <= 0 or not np.isfinite(contour_len):
        return 1.0
    return float(contour_len / end_to_end)

def summarize_chains(centroids, areas, A, D):
    n_components, comp_labels = connected_components(csgraph=A, directed=False, return_labels=True)
    chains = []
    for cid in range(n_components):
        idx = np.where(comp_labels == cid)[0]
        coords = centroids[idx]
        if len(coords) > 1:
            sub_idx = np.ix_(idx, idx)
            A_sub = csr_matrix(A[sub_idx])
            D_sub = D[sub_idx]
            tort = mst_tortuosity(coords, A_sub, D_sub)
        else:
            tort = 1.0
        chains.append({
            "Chain_ID": cid,
            "Num_Particles": int(len(idx)),
            "Avg_Particle_Area_px": float(np.mean(areas[idx])) if len(idx) else np.nan,
            "Tortuosity": float(tort),
            "Centroids": coords,
            "Indices": idx
        })
    chains.sort(key=lambda x: x["Num_Particles"], reverse=True)
    return chains, n_components, comp_labels

def boundary_masks(centroids, axis='left-right', margin=5, H=None, W=None):
    y = centroids[:, 0]
    x = centroids[:, 1]
    if axis == 'left-right':
        touch_a = x <= margin
        touch_b = x >= (W - 1 - margin)
    else:
        touch_a = y <= margin
        touch_b = y >= (H - 1 - margin)
    return touch_a, touch_b

def percolating_components(A, centroids, comp_labels, axis='left-right', margin=5, H=None, W=None):
    percs = []
    touch_a, touch_b = boundary_masks(centroids, axis, margin, H, W)
    ncomp = comp_labels.max() + 1
    for cid in range(ncomp):
        idx = np.where(comp_labels == cid)[0]
        if len(idx) == 0:
            continue
        if np.any(touch_a[idx]) and np.any(touch_b[idx]):
            percs.append(idx)
    return percs

def plot_topology_full(img, centroids, A, perco_components, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap='gray')
    ax.set_title('Topology: full graph (percolation highlighted)')
    ax.axis('off')

    A_dense = A.toarray()
    rows, cols = np.triu_indices_from(A_dense, 1)
    for i, j in zip(rows, cols):
        if A_dense[i, j] == 1:
            p1, p2 = centroids[i], centroids[j]
            ax.plot([p1[1], p2[1]], [p1[0], p2[0]], alpha=0.2, linewidth=0.8)

    ax.scatter(centroids[:,1], centroids[:,0], s=10, c='w', edgecolor='k', linewidth=0.2, alpha=0.5)

    cmap = plt.cm.get_cmap('tab10', max(1, len(perco_components)))
    for k, idx in enumerate(perco_components):
        color = cmap(k)
        sub = A[idx, :][:, idx].toarray()
        for a in range(len(idx)):
            for b in range(a+1, len(idx)):
                if sub[a, b] == 1:
                    p1, p2 = centroids[idx[a]], centroids[idx[b]]
                    ax.plot([p1[1], p2[1]], [p1[0], p2[0]], color=color, linewidth=1.5, alpha=0.9, zorder=3)
        ax.scatter(centroids[idx,1], centroids[idx,0], s=25, color=color, edgecolor='white', linewidth=0.5, zorder=4)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax

def plot_percolation_only(img, centroids, A, perco_components, save_path=None):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(img, cmap='gray')
    ax.set_title('Individual percolating chains')
    ax.axis('off')
    cmap = plt.cm.get_cmap('tab10', max(1, len(perco_components)))
    for k, idx in enumerate(perco_components):
        color = cmap(k)
        sub = A[idx, :][:, idx].toarray()
        for a in range(len(idx)):
            for b in range(a+1, len(idx)):
                if sub[a, b] == 1:
                    p1, p2 = centroids[idx[a]], centroids[idx[b]]
                    ax.plot([p1[1], p2[1]], [p1[0], p2[0]], color=color, linewidth=1.8, zorder=3)
        ax.scatter(centroids[idx,1], centroids[idx,0], s=30, color=color, edgecolor='white', linewidth=0.6, zorder=4)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax

def shortest_boundary_to_boundary_path(centroids, A, axis='left-right', margin=5, H=None, W=None):
    n = A.shape[0]
    if n == 0:
        return None, np.inf
    D_all = squareform(pdist(centroids, 'euclidean'))
    Wm = np.where(A.toarray() > 0, D_all, 0.0)
    Wm = csr_matrix(Wm)
    touch_a, touch_b = boundary_masks(centroids, axis, margin, H, W)
    src_nodes = np.where(touch_a)[0]
    dst_nodes = np.where(touch_b)[0]
    if len(src_nodes) == 0 or len(dst_nodes) == 0:
        return None, np.inf
    dist, pred = dijkstra(Wm, directed=False, indices=src_nodes, return_predecessors=True)
    best_cost = np.inf
    best_pair = None
    best_pred_row = None
    for row_i, src in enumerate(src_nodes):
        for tgt in dst_nodes:
            if np.isfinite(dist[row_i, tgt]) and dist[row_i, tgt] < best_cost:
                best_cost = dist[row_i, tgt]
                best_pair = (src, tgt)
                best_pred_row = pred[row_i]
    if best_pair is None:
        return None, np.inf
    path = []
    cur = best_pair[1]
    while cur != -1 and cur != best_pair[0]:
        path.append(cur)
        cur = best_pred_row[cur]
    if cur == -1:
        return None, np.inf
    path.append(best_pair[0])
    path = path[::-1]
    return np.array(path, dtype=int), float(best_cost)

def plot_shortest_percolation_path(img, centroids, A, perco_components, axis='left-right', margin=5, H=None, W=None, save_path=None):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(img, cmap='gray')
    ax.set_title('Shortest percolation path')
    ax.axis('off')

    if len(perco_components) == 0:
        ax.text(0.5, 0.5, 'No percolation found', transform=ax.transAxes, ha='center', va='center', color='w', fontsize=16)
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig, ax, None

    best_path = None
    best_len = np.inf
    for idx in perco_components:
        subA = A[idx, :][:, idx]
        subC = centroids[idx]
        path_local, cost_local = shortest_boundary_to_boundary_path(subC, subA, axis=axis, margin=margin, H=H, W=W)
        if path_local is not None and cost_local < best_len:
            best_len = cost_local
            best_path = idx[path_local]

    if best_path is None:
        ax.text(0.5, 0.5, 'No boundary to boundary path', transform=ax.transAxes, ha='center', va='center', color='w', fontsize=16)
    else:
        for idx in perco_components:
            sub = A[idx, :][:, idx].toarray()
            for a in range(len(idx)):
                for b in range(a+1, len(idx)):
                    if sub[a, b] == 1:
                        p1, p2 = centroids[idx[a]], centroids[idx[b]]
                        ax.plot([p1[1], p2[1]], [p1[0], p2[0]], alpha=0.25, linewidth=1.0)
        P = best_path
        for i in range(len(P)-1):
            p1, p2 = centroids[P[i]], centroids[P[i+1]]
            ax.plot([p1[1], p2[1]], [p1[0], p2[0]], linewidth=3.0, zorder=5)
        ax.scatter(centroids[P,1], centroids[P,0], s=40, edgecolor='white', linewidth=0.8, zorder=6)

        path_len = 0.0
        for i in range(len(P)-1):
            path_len += np.linalg.norm(centroids[P[i]] - centroids[P[i+1]])
        print(f"Shortest percolation path node count: {len(P)}  length [px]: {path_len:.2f}")
        if PIXEL_SIZE_NM is not None:
            print(f"Shortest path length [nm]: {path_len * PIXEL_SIZE_NM:.2f}")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax, best_path

# ====================== PIPELINE =========================
# 1) Load
try:
    img = robust_read_gray(IMAGE_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Image not found at {IMAGE_PATH}. Update IMAGE_PATH and rerun.")

H, W = img.shape

# 2) Preprocess
base = exposure.equalize_adapthist(img, clip_limit=0.01) if USE_ADAPTHIST else img
blur = filters.gaussian(base, sigma=GAUSS_SIGMA)

# 3) Binary and watershed
binary = otsu_binary(blur, invert=INVERT_FOR_WATERSHED)
dist = ndimage.distance_transform_edt(binary)

rough_markers = measure.label(binary)
fp = FOOTPRINT_PX or auto_footprint_from_regions(rough_markers)
markers = detect_markers(dist, binary, footprint_px=fp)
seg = segmentation.watershed(-dist, markers=markers, mask=binary)

print(f"Particles segmented: {int(seg.max())}")

# 4) Features
centroids, areas, props = region_centroids_areas(seg)
num_particles = len(centroids)
if num_particles == 0:
    raise RuntimeError("No particles detected. Try toggling INVERT_FOR_WATERSHED, GAUSS_SIGMA, or FOOTPRINT_PX.")

# 5) Graph and chains
A, D = build_adjacency(centroids, INTERACTION_DISTANCE_PIXELS)
chains, n_comp, cc_labels = summarize_chains(centroids, areas, A, D)

print("\n--- RESULTS SUMMARY ---")
print(f"Total nanoparticles: {num_particles}")
print(f"Total chains or aggregates: {n_comp}")
print(f"Average particle area [px]: {np.mean(areas):.2f}")
print(f"Marker footprint used: {fp}px")

print("\n--- TOP 5 LARGEST CHAINS ---")
for c in chains[:5]:
    print(f"ID {c['Chain_ID']:>2}  Length={c['Num_Particles']:>3}  Tortuosity={c['Tortuosity']:.3f}")

# 6) Percolation components
perco_components = percolating_components(A, centroids, cc_labels,
                                          axis=PERCOLATION_AXIS, margin=EDGE_MARGIN_PX, H=H, W=W)
print(f"Percolating components found: {len(perco_components)}")

# ====================== FOUR FIGURES =====================
# 1) Original image
fig1, ax1 = plt.subplots(figsize=(10,10))
ax1.imshow(img, cmap='gray')
ax1.set_title('Original image')
ax1.axis('off')
plt.tight_layout()
if SAVE_PREFIX:
    fig1.savefig(f"{SAVE_PREFIX}_1_original.png", dpi=300, bbox_inches='tight')
plt.show()

# 2) Topology with percolation highlighted
fig2, ax2 = plot_topology_full(img, centroids, A, perco_components,
                               save_path=(f"{SAVE_PREFIX}_2_topology.png" if SAVE_PREFIX else None))
plt.tight_layout()
plt.show()

# 3) Individual percolating chains
fig3, ax3 = plot_percolation_only(img, centroids, A, perco_components,
                                  save_path=(f"{SAVE_PREFIX}_3_percolating_chains.png" if SAVE_PREFIX else None))
plt.tight_layout()
plt.show()

# 4) Shortest percolation path
fig4, ax4, best_path = plot_shortest_percolation_path(img, centroids, A, perco_components,
                                                      axis=PERCOLATION_AXIS, margin=EDGE_MARGIN_PX,
                                                      H=H, W=W,
                                                      save_path=(f"{SAVE_PREFIX}_4_shortest_path.png" if SAVE_PREFIX else None))
plt.tight_layout()
plt.show()

# ================== ML FEATURES TABLE ====================
ml_rows = []
for ch in chains:
    row = {
        "Chain_ID": ch["Chain_ID"],
        "Chain_Length_Nodes": ch["Num_Particles"],
        "Tortuosity": ch["Tortuosity"],
        "Avg_Particle_Area_px": ch["Avg_Particle_Area_px"]
    }
    if PIXEL_SIZE_NM is not None:
        row["Interaction_Dist_nm"] = INTERACTION_DISTANCE_PIXELS * PIXEL_SIZE_NM
    ml_rows.append(row)

df = pd.DataFrame(ml_rows)
print("\n--- Data Structure for ML Model ---")
print(df.head())

if SAVE_PREFIX:
    csv_path = f"{SAVE_PREFIX}_chains_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved features to: {csv_path}")

# =================== QUICK TUNING NOTES ==================
# If over-splitting: increase GAUSS_SIGMA a little, increase FOOTPRINT_PX
# If under-splitting: decrease GAUSS_SIGMA a little, decrease FOOTPRINT_PX
# If no percolation but you expect it: increase INTERACTION_DISTANCE_PIXELS a few pixels
# If illumination is uneven: set USE_ADAPTHIST = True
