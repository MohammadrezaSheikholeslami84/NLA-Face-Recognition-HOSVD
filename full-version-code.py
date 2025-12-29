import os
import re
import time
import math
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Sklearn tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

# Image processing
from skimage.feature import local_binary_pattern


# =========================================================
# 1. CONFIGURATION & STYLING
# =========================================================
warnings.filterwarnings("ignore")
np.random.seed(42)

sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": "#222222",
    "axes.linewidth": 1.1,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 18,
    "figure.titleweight": "bold",
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

print("✅ Libraries imported successfully.")


# =========================================================
# 2. PREPROCESSING / HELPERS
# =========================================================

def preprocess_image(img):
    """Standardizes and normalizes an image array to 0-255 range."""
    x = img.astype(np.float32)
    x = x - x.mean()
    x = x / (x.std() + 1e-6)
    x = (x - x.min()) / (x.max() - x.min() + 1e-12)
    return (x * 255.0).astype(np.uint8)


def norm01(x):
    """Normalize array to 0-1 range for visualization."""
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


# =========================================================
# 3. DATASET LOADER
# =========================================================

def load_dataset(data_dir, img_size=(96, 96), max_per_class=None, max_classes=None):
    """
    Loads images from directory structure:
        CroppedYaleB/
            yaleB01/
                yaleB01_P00A-005E-10.pgm
    Returns: images, labels, filenames
    """
    if not os.path.exists(data_dir):
        print(f"❌ Error: Dataset directory '{data_dir}' not found.")
        return np.array([]), np.array([]), np.array([])

    images, labels, filenames = [], [], []
    subjects = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    selected_subjects = subjects[:max_classes] if max_classes else subjects

    print(f"📂 Loading {len(selected_subjects)} classes...")

    for sid, sname in enumerate(selected_subjects):
        spath = os.path.join(data_dir, sname)
        files = sorted([f for f in os.listdir(spath) if f.lower().endswith(('.pgm', '.jpg', '.png'))])
        current_files = files[:max_per_class] if max_per_class else files

        if sid % 5 == 0:
            print(f"  -> Loading subject {sid+1}/{len(selected_subjects)} ({len(current_files)} images)")

        for f in current_files:
            try:
                img = Image.open(os.path.join(spath, f)).convert('L')
                img = img.resize(img_size)
                images.append(preprocess_image(np.array(img)))
                labels.append(sid)
                filenames.append(f)
            except Exception as e:
                print(f"Skipped {f}: {e}")

    print(f"✅ Total Loaded: {len(images)} images from {len(np.unique(labels))} subjects.")
    return np.array(images), np.array(labels), np.array(filenames)


# =========================================================
# 4. VISUALIZATION FUNCTIONS
# =========================================================

def show_predictions_with_recon(X_test, y_test, y_pred, X_train, y_train, reconstruct_fn, method_name="Method", max_show=5, save_path=None):
    """
    Shows 3 columns:
    1. Test Image (Original)
    2. Reconstructed Image
    3. Predicted Class Representative
    """
    idx_correct = np.where(y_test == y_pred)[0]
    idx_wrong = np.where(y_test != y_pred)[0]

    # Mix correct and wrong predictions
    n_correct = min(len(idx_correct), max_show // 2 + 1)
    n_wrong = min(len(idx_wrong), max_show - n_correct)
    
    selected = []
    if len(idx_correct) > 0: selected.extend(idx_correct[:n_correct])
    if len(idx_wrong) > 0: selected.extend(idx_wrong[:n_wrong])
    selected = np.array(selected, dtype=int)

    if len(selected) == 0: return

    # Helper to find a representative image for a class (e.g., first image of that class in train set)
    class_rep = {}
    for cls in np.unique(y_pred[selected]):
        idx_cls = np.where(y_train == cls)[0]
        class_rep[cls] = X_train[idx_cls[0]] if len(idx_cls) > 0 else np.zeros_like(X_test[0])

    n = len(selected)
    fig = plt.figure(figsize=(12, 3.5 * n))
    
    for r, i in enumerate(selected):
        img = X_test[i]
        true_lbl = y_test[i]
        pred_lbl = y_pred[i]
        
        # Reconstruct the specific test image
        rec_img = reconstruct_fn(img)

        is_correct = (true_lbl == pred_lbl)
        color = "#27ae60" if is_correct else "#c0392b"
        status = "CORRECT" if is_correct else "WRONG"

        # --- Column 1: Test Image ---
        ax1 = fig.add_subplot(n, 3, 3*r + 1)
        ax1.imshow(img, cmap="gray")
        # Add colored border
        for spine in ax1.spines.values():
            spine.set_color(color)
            spine.set_linewidth(3)
        ax1.set_title(f"Test Input\nTrue: S{true_lbl+1}", fontsize=11, fontweight="bold")
        ax1.set_ylabel(f"Example {r+1}", fontsize=12, fontweight='bold', labelpad=10)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # --- Column 2: Reconstruction ---
        ax2 = fig.add_subplot(n, 3, 3*r + 2)
        ax2.imshow(norm01(rec_img), cmap="gray")
        ax2.set_title("Reconstruction", fontsize=11)
        ax2.axis("off")

        # --- Column 3: Predicted Class ---
        ax3 = fig.add_subplot(n, 3, 3*r + 3)
        ax3.imshow(class_rep[pred_lbl], cmap="gray")
        ax3.set_title(f"Predicted Identity\nPred: S{pred_lbl+1} ({status})", fontsize=11, fontweight="bold", color=color)
        ax3.axis("off")

    plt.suptitle(f"{method_name}: Input vs Reconstruction vs Prediction", fontsize=16, y=1.0) # Adjusted y
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print("✅ Saved", save_path)
    plt.show()
    plt.close()


def show_predictions_with_recon_eigenfaces(
    X_test, y_test, y_pred,
    X_train, y_train,
    eigen_model,
    method_name="EigenFaces",
    max_show=6,
    save_path=None,
    show=True
):
    """
    Shows 3 columns for EigenFaces:
    1) Test image (original)
    2) EigenFaces reconstruction
    3) Representative image of predicted class (from training set)

    - Mixes correct and wrong predictions
    - Adds green/red border for correctness
    - Saves the figure if save_path is provided
    """

    if eigen_model.mean is None or eigen_model.W is None:
        raise ValueError("❌ EigenFaces model must be fitted before calling this function.")

    idx_correct = np.where(y_test == y_pred)[0]
    idx_wrong   = np.where(y_test != y_pred)[0]

    # Select mix of correct + wrong
    n_correct = min(len(idx_correct), max_show // 2 + 1)
    n_wrong   = min(len(idx_wrong), max_show - n_correct)

    selected = []
    if len(idx_correct) > 0:
        selected.extend(idx_correct[:n_correct])
    if len(idx_wrong) > 0:
        selected.extend(idx_wrong[:n_wrong])

    selected = np.array(selected, dtype=int)
    if len(selected) == 0:
        print("⚠️ No samples selected to visualize.")
        return

    # Representative image for each class (first training image of that class)
    class_rep = {}
    for cls in np.unique(y_pred[selected]):
        idx_cls = np.where(y_train == cls)[0]
        class_rep[cls] = X_train[idx_cls[0]] if len(idx_cls) > 0 else np.zeros_like(X_test[0])

    n = len(selected)
    fig = plt.figure(figsize=(12, 3.5 * n))

    for r, i in enumerate(selected):
        img = X_test[i]
        true_lbl = y_test[i]
        pred_lbl = y_pred[i]

        # Reconstruction using eigenfaces
        rec_img = eigen_model.reconstruct(img)

        is_correct = (true_lbl == pred_lbl)
        color = "#27ae60" if is_correct else "#c0392b"
        status = "CORRECT" if is_correct else "WRONG"

        # --- Column 1: Test Image ---
        ax1 = fig.add_subplot(n, 3, 3*r + 1)
        ax1.imshow(img, cmap="gray")
        for spine in ax1.spines.values():
            spine.set_color(color)
            spine.set_linewidth(3)

        ax1.set_title(f"Test Input\nTrue: S{true_lbl+1}", fontsize=11, fontweight="bold")
        ax1.set_ylabel(f"Example {r+1}", fontsize=12, fontweight="bold", labelpad=10)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # --- Column 2: Reconstruction ---
        ax2 = fig.add_subplot(n, 3, 3*r + 2)
        ax2.imshow(norm01(rec_img), cmap="gray")
        ax2.set_title("EigenFaces Reconstruction", fontsize=11)
        ax2.axis("off")

        # --- Column 3: Predicted Class Representative ---
        ax3 = fig.add_subplot(n, 3, 3*r + 3)
        ax3.imshow(class_rep[pred_lbl], cmap="gray")
        ax3.set_title(
            f"Predicted Identity\nPred: S{pred_lbl+1} ({status})",
            fontsize=11,
            fontweight="bold",
            color=color
        )
        ax3.axis("off")

    plt.suptitle(f"{method_name}: Input vs Reconstruction vs Prediction", fontsize=16, y=1.0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print("✅ Saved", save_path)

    if show:
        plt.show()
    plt.close()


def plot_accuracy_comparison(acc_dict, out="comparison_accuracy.png"):
    methods = list(acc_dict.keys())
    accs = list(acc_dict.values())
    
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("viridis", n_colors=len(methods))
    bars = plt.bar(methods, accs, color=palette, edgecolor='black', alpha=0.85, width=0.6)

    plt.ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
    plt.title("Performance Comparison: Accuracy", fontsize=15, pad=20)
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{acc:.1f}%",
                 ha="center", va="bottom", fontsize=12, fontweight='bold', color='#333')
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print("✅ Saved", out)
    plt.show()


def plot_time_comparison(time_dict, out="comparison_time.png"):
    methods = list(time_dict.keys())
    times = list(time_dict.values())

    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("Reds", n_colors=len(methods))[::-1]
    bars = plt.bar(methods, times, color=palette, edgecolor='black', alpha=0.85, width=0.6)

    plt.ylabel("Execution Time (seconds)", fontsize=12, fontweight='bold')
    plt.title("Performance Comparison: Time", fontsize=15, pad=20)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, t in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(times)*0.02),
                 f"{t:.2f}s", ha="center", va="bottom", fontsize=11, color='#333')

    sns.despine()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print("✅ Saved", out)
    plt.show()


def save_single_reconstruction(X, reconstruct_fn, idx=0, title="Reconstruction", out="reconstruction.png"):
    """
    Save a side-by-side original vs reconstruction for one image.
    """
    if idx is None:
        idx = np.random.randint(0, len(X))

    img = X[idx]
    rec = reconstruct_fn(img)
    
    # Calculate MSE for title
    mse = np.mean((img.astype(float) - rec.astype(float))**2)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(norm01(rec), cmap="gray")
    axes[1].axis("off")

    plt.suptitle(f"{title} (index={idx})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print("✅ Saved", out)
    plt.show()
    plt.close()


def show_predictions_simple(X_test, y_test, y_pred, X_train, y_train, method_name="Method", max_show=8, save_path=None):
    """
    Shows some correct and wrong predictions.
    """
    idx_correct = np.where(y_test == y_pred)[0]
    idx_wrong = np.where(y_test != y_pred)[0]

    n_correct = min(len(idx_correct), max_show // 2)
    n_wrong = min(len(idx_wrong), max_show - n_correct)
    
    selected = []
    if len(idx_correct) > 0: selected.extend(idx_correct[:n_correct])
    if len(idx_wrong) > 0: selected.extend(idx_wrong[:n_wrong])
    selected = np.array(selected, dtype=int)

    if len(selected) == 0: return

    # Helper to find a representative image for a class
    class_rep = {}
    for cls in np.unique(y_pred[selected]):
        idx_cls = np.where(y_train == cls)[0]
        class_rep[cls] = X_train[idx_cls[0]] if len(idx_cls) > 0 else np.zeros_like(X_test[0])

    n = len(selected)
    rows = int(np.ceil(n / 2))
    fig = plt.figure(figsize=(10, 3.0 * rows))

    for r, i in enumerate(selected):
        img = X_test[i]
        true_lbl = y_test[i]
        pred_lbl = y_pred[i]
        ok = (true_lbl == pred_lbl)
        color = "#27ae60" if ok else "#c0392b"

        # Query Image
        ax1 = fig.add_subplot(rows, 4, 2*r + 1)
        ax1.imshow(img, cmap="gray")
        for spine in ax1.spines.values():
            spine.set_color(color)
            spine.set_linewidth(3)
        ax1.set_title(f"Test: S{true_lbl+1}", fontsize=10, fontweight="bold")
        ax1.axis("off")

        # Predicted Class Representative
        ax2 = fig.add_subplot(rows, 4, 2*r + 2)
        ax2.imshow(class_rep[pred_lbl], cmap="gray")
        ax2.set_title(f"Pred: S{pred_lbl+1}", fontsize=10, fontweight="bold", color=color)
        ax2.axis("off")

    plt.suptitle(f"{method_name} Examples", fontsize=16, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print("✅ Saved", save_path)
    plt.show()
    plt.close()

def extract_and_save_eigenfaces_bases(eigen_model, H, W, n_show=16,
                                      save_img="eigenfaces_bases.png",
                                      save_npy="eigenfaces_bases.npy",
                                      show=True):
    """
    Extract and save EigenFaces bases from fitted EigenFaces model.

    Saves:
      - Image grid (png)
      - Raw bases array (npy) with shape (n_show, H, W)
    """
    if eigen_model.W is None:
        raise ValueError("❌ EigenFaces model must be fitted before extracting bases.")

    n_show = min(n_show, eigen_model.W.shape[1])

    bases = []
    for i in range(n_show):
        face = eigen_model.W[:, i].reshape(H, W)
        bases.append(face)

    bases = np.array(bases, dtype=np.float32)  # (n_show, H, W)

    # Save npy
    np.save(save_npy, bases)
    print(f"✅ Saved EigenFaces bases array: {save_npy} | shape={bases.shape}")

    # Plot grid
    cols = int(np.ceil(np.sqrt(n_show)))
    rows = int(np.ceil(n_show / cols))

    plt.figure(figsize=(2.2 * cols, 2.2 * rows))
    for i in range(n_show):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(norm01(bases[i]), cmap="gray")
        ax.set_title(f"EF {i+1}", fontsize=9)
        ax.axis("off")

    plt.suptitle("EigenFaces Bases", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_img, dpi=200, bbox_inches="tight")
    print(f"✅ Saved EigenFaces bases image: {save_img}")

    if show:
        plt.show()
    plt.close()

    return bases


def extract_and_save_tensorfaces_bases(tensor_model, H, W, n_show=16,
                                       save_img="tensorfaces_bases.png",
                                       save_npy="tensorfaces_bases.npy",
                                       show=True):
    """
    Extract and save TensorFacesLite pixel bases (U_p columns).

    Saves:
      - Image grid (png)
      - Raw bases array (npy) with shape (n_show, H, W)
    """
    if tensor_model.U_p is None:
        raise ValueError("❌ TensorFacesLite model must be fitted before extracting bases.")

    n_show = min(n_show, tensor_model.U_p.shape[1])

    bases = []
    for i in range(n_show):
        basis_img = tensor_model.U_p[:, i].reshape(H, W)
        bases.append(basis_img)

    bases = np.array(bases, dtype=np.float32)  # (n_show, H, W)

    # Save npy
    np.save(save_npy, bases)
    print(f"✅ Saved TensorFaces bases array: {save_npy} | shape={bases.shape}")

    # Plot grid
    cols = int(np.ceil(np.sqrt(n_show)))
    rows = int(np.ceil(n_show / cols))

    plt.figure(figsize=(2.2 * cols, 2.2 * rows))
    for i in range(n_show):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(norm01(bases[i]), cmap="gray")
        ax.set_title(f"TF {i+1}", fontsize=9)
        ax.axis("off")

    plt.suptitle("TensorFacesLite Bases (U_p)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_img, dpi=200, bbox_inches="tight")
    print(f"✅ Saved TensorFaces bases image: {save_img}")

    if show:
        plt.show()
    plt.close()

    return bases


# =========================================================
# 5. MODEL CLASSES
# =========================================================

# --- A. EIGENFACES ---
class EigenFaces:
    def __init__(self, n_components=100):
        self.k = n_components
        self.mean = None
        self.W = None
        self.class_means = None

    def fit(self, images, labels, num_classes):
        X = images.reshape(len(images), -1).astype(np.float32)
        self.mean = X.mean(axis=0)
        Xc = X - self.mean
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.W = Vt.T[:, :self.k]
        Z = Xc @ self.W
        
        self.class_means = np.zeros((num_classes, self.k))
        for c in range(num_classes):
            if np.any(labels == c):
                self.class_means[c] = Z[labels==c].mean(axis=0)

    def predict(self, images):
        X = images.reshape(len(images), -1).astype(np.float32)
        Z = (X - self.mean) @ self.W
        preds = []
        for z in Z:
            d = np.linalg.norm(self.class_means - z, axis=1)
            preds.append(np.argmin(d))
        return np.array(preds)

    def reconstruct(self, img):
        x = img.reshape(1, -1).astype(np.float32)
        z = (x - self.mean) @ self.W
        xr = self.mean + z @ self.W.T
        return xr.reshape(img.shape)


# --- B. LBP ---
class LBPClassifier:
    def __init__(self, P=8, R=1, grid_x=8, grid_y=8, num_bins=59):
        self.P, self.R = P, R
        self.gx, self.gy = grid_x, grid_y
        self.num_bins = num_bins
        self.class_means = None

    def extract_lbp(self, img):
        lbp = local_binary_pattern(img, self.P, self.R, method="uniform")
        h, w = img.shape
        hx, wy = h // self.gx, w // self.gy
        features = []
        for i in range(self.gx):
            for j in range(self.gy):
                block = lbp[i*hx:(i+1)*hx, j*wy:(j+1)*wy]
                hist, _ = np.histogram(block.ravel(), bins=self.num_bins, range=(0, self.num_bins), density=True)
                features.append(hist)
        return np.concatenate(features)

    def fit(self, images, labels, num_classes):
        feats = np.array([self.extract_lbp(img) for img in images])
        self.class_means = np.zeros((num_classes, feats.shape[1]))
        for c in range(num_classes):
            if np.any(labels == c):
                self.class_means[c] = feats[labels == c].mean(axis=0)

    def predict(self, images):
        preds = []
        for img in images:
            feat = self.extract_lbp(img)
            d = np.linalg.norm(self.class_means - feat, axis=1)
            preds.append(np.argmin(d))
        return np.array(preds)


# --- C. FISHERFACES ---
class FisherFaces:
    def __init__(self, pca_components=100):
        self.pca_components = pca_components
        self.pca = None
        self.lda = None

    def fit(self, images, labels):
        X = images.reshape(len(images), -1).astype(np.float32)
        self.pca = PCA(n_components=self.pca_components, whiten=True)
        Xp = self.pca.fit_transform(X)
        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(Xp, labels)

    def predict(self, images):
        X = images.reshape(len(images), -1).astype(np.float32)
        Xp = self.pca.transform(X)
        return self.lda.predict(Xp)


# --- D. TENSORFACES LOGIC ---

def parse_light(fn):
    """Parses light condition from filename (CroppedYaleB format)."""
    base = os.path.splitext(os.path.basename(fn))[0]
    m = re.search(r"(A[+-]\d{3}E[+-]\d{2})", base)
    if m: return m.group(1)
    return "unknown"

def build_subject_light_tensor(images, labels, filenames):
    """Builds (Subject, Light, Pixel) tensor."""
    H, W = images.shape[1:]
    Npix = H * W
    subjects = sorted(np.unique(labels))
    lights = sorted({parse_light(f) for f in filenames})
    
    subj_to_i = {s: i for i, s in enumerate(subjects)}
    light_to_j = {l: j for j, l in enumerate(lights)}

    D = np.zeros((len(subjects), len(lights), Npix), dtype=np.float32)
    count = np.zeros((len(subjects), len(lights)), dtype=np.int32)

    for img, lab, fn in zip(images, labels, filenames):
        i = subj_to_i[lab]
        j = light_to_j[parse_light(fn)]
        D[i, j] += img.reshape(-1).astype(np.float32) / 255.0
        count[i, j] += 1

    # Average duplicates
    for i in range(len(subjects)):
        for j in range(len(lights)):
            if count[i, j] > 1:
                D[i, j] /= count[i, j]

    return D, subjects, lights, (H, W)

def unfold(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))

def hosvd_3d(D, ranks):
    """Higher-Order SVD on 3D tensor."""
    t0 = time.time()
    U = []
    for mode in range(3):
        A = unfold(D, mode)
        U_m, _, _ = np.linalg.svd(A, full_matrices=False)
        U.append(U_m[:, :ranks[mode]])

    # Calculate core
    # core = D x1 U1.T x2 U2.T x3 U3.T
    core = D.copy()
    core = np.moveaxis(np.tensordot(U[0].T, core, axes=(1, 0)), 0, 0)
    core = np.moveaxis(np.tensordot(U[1].T, core, axes=(1, 1)), 0, 1)
    core = np.moveaxis(np.tensordot(U[2].T, core, axes=(1, 2)), 0, 2)
    
    return core, U, time.time() - t0


class TensorFacesLiteRealLight:
    """
    TensorFaces implementation using HOSVD + Ridge Regression per Light Source.
    Includes reconstruction capability.
    """
    def __init__(self, H, W, ranks=(30, 10, 200), use_ridge=True, lam=1e-2):
        self.H, self.W = H, W
        self.rS, self.rL, self.rP = ranks
        self.use_ridge = use_ridge
        self.lam = lam
        
        self.U_s = None
        self.U_p = None
        self.mean_pix = None
        self.Mjt_list = None
        self.subjects = None

    def fit(self, D, subjects, lights):
        S, L, Npix = D.shape
        self.subjects = subjects
        
        # Mean centering
        self.mean_pix = D.reshape(-1, Npix).mean(axis=0).astype(np.float32)

        # HOSVD to get bases
        actual_rS = min(self.rS, S)
        actual_rL = min(self.rL, L)
        actual_rP = min(self.rP, Npix)
        
        core, (U_s, U_l, U_p), t = hosvd_3d(D, ranks=[actual_rS, actual_rL, actual_rP])
        self.U_s = U_s.astype(np.float32)  # (S, rS)
        self.U_p = U_p.astype(np.float32)  # (Npix, rP)

        # Precompute Matrices per Light Source
        # M_j = (U_s^T @ D[:, j, :] @ U_p)^T
        self.Mjt_list = []
        for j in range(L):
            Dj = D[:, j, :].astype(np.float32)  # (S, Npix)
            Dsub = self.U_s.T @ Dj @ self.U_p
            self.Mjt_list.append(Dsub.T) # Store (rP, rS) for solving

    def predict_one(self, img):
        x = img.reshape(-1).astype(np.float32) / 255.0
        x = x - self.mean_pix
        
        # Project to Pixel Subspace
        y = self.U_p.T @ x  # (rP,)

        best_j, best_a, best_res = None, None, np.inf

        # Solve for coefficients 'a' (subject weights) for each light condition
        for j, Mjt in enumerate(self.Mjt_list):
            if self.use_ridge:
                # Ridge: (M^T M + lam I)^-1 M^T y
                AtA = Mjt.T @ Mjt
                AtA.flat[::AtA.shape[0]+1] += self.lam
                rhs = Mjt.T @ y
                a = np.linalg.solve(AtA, rhs)
            else:
                a, *_ = np.linalg.lstsq(Mjt, y, rcond=None)
            
            # Residual in subspace
            res = np.linalg.norm(y - Mjt @ a)

            if res < best_res:
                best_res = res
                best_j = j
                best_a = a

        
        scores = self.U_s @ best_a
        pred_idx = int(np.argmax(scores))
        return self.subjects[pred_idx]

    def predict(self, X_test):
        preds = []
        # No tqdm here to keep output clean during analysis
        for img in X_test:
            preds.append(self.predict_one(img))
        return np.array(preds)

    def reconstruct(self, img):
        """
        Reconstructs image using Pixel Basis (U_p).
        x_rec = mean + (x_centered @ U_p) @ U_p.T
        """
        x = img.reshape(-1).astype(np.float32) / 255.0
        xc = x - self.mean_pix
        
        # Project and back-project
        coeff = xc @ self.U_p  # (rP,)
        xr = self.mean_pix + coeff @ self.U_p.T
        
        xr = xr.reshape(self.H, self.W)
        return (np.clip(xr, 0, 1) * 255.0).astype(np.uint8)


# =========================================================
# 6. HYPERPARAMETER ANALYSIS FUNCTIONS
# =========================================================

def eigenfaces_hyperparams_analysis(eigen_model, X_train, y_train, X_test, y_test, ks, num_classes):
    print("\n" + "="*70)
    print("🔎 RUNNING EIGENFACES HYPERPARAMETER ANALYSIS (Accuracy & Time)")
    print("="*70)
    
    accs = []
    times = []
    original_k = eigen_model.k

    print(f"| {'k (Components)':^15} | {'Accuracy':^15} | {'Time (s)':^15} |")
    print("-" * 55)

    for k in ks:
        eigen_model.k = int(k)
        
        t_start = time.time()
        eigen_model.fit(X_train, y_train, num_classes)
        preds = eigen_model.predict(X_test)
        t_end = time.time()
        
        duration = t_end - t_start
        acc = accuracy_score(y_test, preds) * 100
        
        accs.append(acc)
        times.append(duration)
        print(f"| {k:^15} | {acc:^14.2f}% | {duration:^15.4f} |")

    eigen_model.k = original_k # restore

    # Plotting Dual Axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Number of Eigenfaces (k)', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', color=color, fontsize=12, fontweight='bold')
    ax1.plot(ks, accs, marker="o", markersize=8, linewidth=2.5, color=color, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Execution Time (seconds)', color=color, fontsize=12, fontweight='bold')
    ax2.plot(ks, times, marker="s", markersize=8, linewidth=2.5, color=color, linestyle='--', label='Time')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(False) # Turn off grid for second axis to avoid clutter

    plt.title("EigenFaces: Accuracy vs. Time tradeoff", fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig("analysis_eigenfaces_k_time.png", dpi=200)
    print("\n✅ Saved analysis_eigenfaces_k_time.png")
    plt.show()

def tensorfaces_single_param_analysis(D_train, subjects, lights, X_test, y_test, H, W,
                                      param_name, param_values,
                                      fixed_rS=30, fixed_rL=10, fixed_rP=150, fixed_lam=0.01,
                                      save_prefix="tensor_param", show=True):
    """
    Analyze sensitivity of TensorFacesLiteRealLight to one parameter:
      param_name in {"rS", "rL", "rP", "lam"}

    Prints metrics for each value in terminal and saves plots.
    """
    accs = []
    times = []

    valid_params = {"rS", "rL", "rP", "lam"}
    if param_name not in valid_params:
        raise ValueError(f"param_name must be one of: {sorted(list(valid_params))}")

    print("\n" + "="*75)
    print(f"📌 TensorFacesLite Parameter Sensitivity: {param_name}")
    print("="*75)
    print(f"Fixed: rS={fixed_rS}, rL={fixed_rL}, rP={fixed_rP}, lam={fixed_lam}")
    print("-" * 75)
    print(f"| {param_name:^10} | {'Accuracy (%)':^15} | {'Time (s)':^15} |")
    print("-" * 50)

    for v in param_values:

        # default fixed values
        rS, rL, rP, lam = fixed_rS, fixed_rL, fixed_rP, fixed_lam

        # update only the chosen param
        if param_name == "rS":
            rS = int(v)
        elif param_name == "rL":
            rL = int(v)
        elif param_name == "rP":
            rP = int(v)
        elif param_name == "lam":
            lam = float(v)

        start = time.time()
        model = TensorFacesLiteRealLight(H, W, ranks=(rS, rL, rP), lam=lam)
        model.fit(D_train, subjects, lights)
        preds = model.predict(X_test)
        duration = time.time() - start

        acc = accuracy_score(y_test, preds) * 100

        accs.append(acc)
        times.append(duration)

        print(f"| {str(v):^10} | {acc:^15.2f} | {duration:^15.4f} |")

    # ===============================
    # Plot Accuracy
    # ===============================
    plt.figure(figsize=(8, 5))
    plt.plot(param_values, accs, marker="o", linewidth=2.5)
    plt.xlabel(param_name)
    plt.ylabel("Accuracy (%)")
    plt.title(f"TensorFacesLite: Accuracy vs {param_name}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    out1 = f"{save_prefix}_{param_name}_acc.png"
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    print(f"\n✅ Saved: {out1}")
    if show:
        plt.show()
    plt.close()

    # ===============================
    # Plot Time
    # ===============================
    plt.figure(figsize=(8, 5))
    plt.plot(param_values, times, marker="s", linewidth=2.5)
    plt.xlabel(param_name)
    plt.ylabel("Time (seconds)")
    plt.title(f"TensorFacesLite: Time vs {param_name}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    out2 = f"{save_prefix}_{param_name}_time.png"
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    print(f"✅ Saved: {out2}")
    if show:
        plt.show()
    plt.close()

    return accs, times


def show_and_save_predictions_with_reconstruction(
    X_test, y_test, y_pred,
    X_train, y_train,
    reconstruct_fn,
    method_name="Method",
    max_show=6,
    save_path=None,
    show=True,
    layout="input_pred_recon",   # or "input_recon_pred"
    use_norm_for_recon=True      # normalize recon output for visualization
):

    idx_correct = np.where(y_test == y_pred)[0]
    idx_wrong = np.where(y_test != y_pred)[0]

    # choose a mix: half correct + half wrong
    n_correct = min(len(idx_correct), max_show // 2)
    n_wrong = min(len(idx_wrong), max_show - n_correct)

    selected_indices = []
    if n_correct > 0:
        selected_indices.extend(idx_correct[:n_correct])
    if n_wrong > 0:
        selected_indices.extend(idx_wrong[:n_wrong])

    selected_indices = np.array(selected_indices, dtype=int)
    n = len(selected_indices)
    if n == 0:
        print("⚠️ No samples selected for visualization.")
        return

    # Representative images for predicted classes (from training set)
    class_representatives = {}
    for cls in np.unique(y_pred[selected_indices]):
        idx_cls = np.where(y_train == cls)[0]
        class_representatives[cls] = X_train[idx_cls[0]] if len(idx_cls) > 0 else np.zeros_like(X_test[0])

    fig = plt.figure(figsize=(12, 3.8 * n))

    for r, i in enumerate(selected_indices):
        img = X_test[i]
        true_lbl = int(y_test[i])
        pred_lbl = int(y_pred[i])

        is_correct = (true_lbl == pred_lbl)
        color = "#27ae60" if is_correct else "#c0392b"
        status_text = "✓ MATCH" if is_correct else "✗ MISMATCH"

        # reconstruction
        rec = reconstruct_fn(img)
        if use_norm_for_recon:
            rec_show = norm01(rec)
        else:
            rec_show = rec

        # representative
        rep_img = class_representatives.get(pred_lbl, np.zeros_like(img))

        # --- column order ---
        if layout == "input_pred_recon":
            col2_img, col2_title = rep_img, f"Predicted Class Sample\n(Pred: S{pred_lbl+1})"
            col3_img, col3_title = rec_show, f"{method_name}\nReconstruction"
        elif layout == "input_recon_pred":
            col2_img, col2_title = rec_show, f"{method_name}\nReconstruction"
            col3_img, col3_title = rep_img, f"Predicted Class Sample\n(Pred: S{pred_lbl+1})"
        else:
            raise ValueError("layout must be either 'input_pred_recon' or 'input_recon_pred'")

        # ------------------------------
        # Col 1: Input Image
        # ------------------------------
        ax1 = fig.add_subplot(n, 3, 3*r + 1)
        ax1.imshow(img, cmap="gray")
        for spine in ax1.spines.values():
            spine.set_color(color)
            spine.set_linewidth(3)

        ax1.set_title(f"Test Input\n(True: S{true_lbl+1})", fontsize=10, fontweight="bold", pad=8)
        ax1.set_ylabel(status_text, fontsize=12, fontweight="bold", color=color, labelpad=10)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # ------------------------------
        # Col 2
        # ------------------------------
        ax2 = fig.add_subplot(n, 3, 3*r + 2)
        ax2.imshow(col2_img, cmap="gray")
        ax2.set_title(col2_title, fontsize=10, fontweight="bold", color="#444", pad=8)

        # marker if wrong and the middle column is the predicted sample
        if (not is_correct) and (layout == "input_pred_recon"):
            ax2.text(
                5, 12, "?", color="white", fontsize=20, fontweight="bold",
                bbox=dict(facecolor=color, alpha=0.8, edgecolor="none", boxstyle="circle")
            )

        ax2.axis("off")

        # ------------------------------
        # Col 3
        # ------------------------------
        ax3 = fig.add_subplot(n, 3, 3*r + 3)
        ax3.imshow(col3_img, cmap="gray")
        ax3.set_title(col3_title, fontsize=10, fontweight="bold", color="#555", pad=8)

        # marker if wrong and the predicted sample is last column
        if (not is_correct) and (layout == "input_recon_pred"):
            ax3.text(
                5, 12, "?", color="white", fontsize=20, fontweight="bold",
                bbox=dict(facecolor=color, alpha=0.8, edgecolor="none", boxstyle="circle")
            )

        ax3.axis("off")

    plt.suptitle(f"{method_name} Error Analysis", fontsize=16, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print("✅ Saved", save_path)

    if show:
        plt.show()

    plt.close()

def tensorfaces_hyperparams_analysis(D_train, subjects, lights, X_test, y_test, H, W):
    print("\n" + "="*60)
    print("🔎 RUNNING TENSORFACES HYPERPARAMETER ANALYSIS")
    print("="*60)

    # Fixed base parameters
    fixed_rS = 30
    fixed_rL = 10
    fixed_rP = 150
    fixed_lam = 0.01

    # ==========================================
    # 1. Effect of Pixel Rank (rP)
    # ==========================================
    rP_values = [50, 100, 150, 200, 300, 400]
    acc_rP = []
    time_rP = []

    print(f"\n🔹 PART 1: Effect of Pixel Rank (rP)")
    print(f"   (Fixed: rS={fixed_rS}, rL={fixed_rL}, lambda={fixed_lam})")
    print("-" * 65)
    print(f"| {'rP Value':^12} | {'Accuracy':^15} | {'Time (s)':^15} | {'Status':^10} |")
    print("-" * 65)

    for rp in rP_values:
        t_start = time.time()
        
        # Training & Inference
        model = TensorFacesLiteRealLight(H, W, ranks=(fixed_rS, fixed_rL, rp), lam=fixed_lam)
        model.fit(D_train, subjects, lights)
        preds = model.predict(X_test)
        
        t_end = time.time()
        duration = t_end - t_start
        acc = accuracy_score(y_test, preds) * 100
        
        acc_rP.append(acc)
        time_rP.append(duration)

        # Print Row
        print(f"| {rp:^12} | {acc:^14.2f}% | {duration:^15.4f} | {'Done':^10} |")

    print("-" * 65)

    # Plot rP Results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    sns.lineplot(x=rP_values, y=acc_rP, marker="s", linewidth=2.5, color="#2ecc71", ax=ax1)
    ax1.set_ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
    ax1.set_title(f"Effect of Pixel Rank ($r_P$)\n(Fixed $r_S$={fixed_rS})", fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    sns.lineplot(x=rP_values, y=time_rP, marker="o", linewidth=2.5, color="#34495e", ax=ax2)
    ax2.set_ylabel("Time (seconds)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Pixel Rank ($r_P$)", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("analysis_tensor_rP_metrics.png", dpi=300)
    print("✅ Saved analysis_tensor_rP_metrics.png")
    plt.show()


    # ==========================================
    # 2. Effect of Subject Rank (rS)
    # ==========================================
    max_subjects = len(np.unique(subjects))
    rS_values = [5, 10, 15, 20, 25, 30]
    rS_values = [r for r in rS_values if r <= max_subjects]
    
    acc_rS = []
    time_rS = []

    print(f"\n🔹 PART 2: Effect of Subject Rank (rS)")
    print(f"   (Fixed: rP={fixed_rP}, rL={fixed_rL}, lambda={fixed_lam})")
    print("-" * 65)
    print(f"| {'rS Value':^12} | {'Accuracy':^15} | {'Time (s)':^15} | {'Status':^10} |")
    print("-" * 65)

    for rs in rS_values:
        t_start = time.time()
        
        # Training & Inference
        model = TensorFacesLiteRealLight(H, W, ranks=(rs, fixed_rL, fixed_rP), lam=fixed_lam)
        model.fit(D_train, subjects, lights)
        preds = model.predict(X_test)
        
        t_end = time.time()
        duration = t_end - t_start
        acc = accuracy_score(y_test, preds) * 100
        
        acc_rS.append(acc)
        time_rS.append(duration)

        # Print Row
        print(f"| {rs:^12} | {acc:^14.2f}% | {duration:^15.4f} | {'Done':^10} |")

    print("-" * 65)

    # Plot rS Results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    sns.lineplot(x=rS_values, y=acc_rS, marker="D", linewidth=2.5, color="#e74c3c", ax=ax1)
    ax1.set_ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
    ax1.set_title(f"Effect of Subject Rank ($r_S$)\n(Fixed $r_P$={fixed_rP})", fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    sns.lineplot(x=rS_values, y=time_rS, marker="o", linewidth=2.5, color="#34495e", ax=ax2)
    ax2.set_ylabel("Time (seconds)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Subject Rank ($r_S$)", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("analysis_tensor_rS_metrics.png", dpi=300)
    print("✅ Saved analysis_tensor_rS_metrics.png")
    plt.show()


def show_preprocessing_steps(original_img, mean_face, H, W, save_path=None, show=True):
    """Shows original, mean face, and centered image."""
    mean_img = mean_face.reshape(H, W) if mean_face.ndim == 1 else mean_face
    centered = original_img.astype(np.float32) - mean_img.astype(np.float32)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    cmap = "gray"

    axes[0].imshow(original_img, cmap=cmap)
    axes[0].set_title("Original Input", fontsize=14, pad=10)
    
    axes[1].imshow(mean_img, cmap=cmap)
    axes[1].set_title("Global Mean Face", fontsize=14, pad=10)
    
    axes[2].imshow(norm01(centered), cmap=cmap)
    axes[2].set_title("Zero-Mean (Centered)", fontsize=14, pad=10)

    for ax in axes:
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#ddd')

    plt.suptitle("Preprocessing Pipeline", y=0.98, fontsize=16, color="#333")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print("✅ Saved", save_path)
    if show:
        plt.show()
    plt.close()


def show_diff_subjects(X_img, y, subject_ids=None, save_path=None, max_cols=7):
    """Shows one representative image from multiple subjects."""
    if subject_ids is None:
        subject_ids = np.unique(y)
    n = len(subject_ids)
    n_cols = min(max_cols, n)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2.2 * n_rows), constrained_layout=True)
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, sid in zip(axes, subject_ids):
        idx = np.where(y == sid)[0][0]
        ax.imshow(X_img[idx], cmap="gray")
        ax.set_title(f"Subject {sid + 1}", fontsize=10, fontweight='bold', color="#444")
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Dataset Subjects Overview", fontsize=16, y=1.02)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print("✅ Saved", save_path)
    plt.close()


def show_same_subject_diff_illumination(X_img, y, subject_id, n_lights=16, save_path=None):
    """Shows multiple images from the same subject (first N samples)."""
    idxs = np.where(y == subject_id)[0][:n_lights]
    cols = 8
    rows = int(np.ceil(len(idxs) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2.2*rows))
    axes = axes.flatten()

    for ax, idx in zip(axes, idxs):
        ax.imshow(X_img[idx], cmap="gray")
        ax.axis("off")

    for ax in axes[len(idxs):]:
        ax.axis("off")

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.suptitle(f"Subject {subject_id+1}: Illumination Variations", fontsize=16, y=0.95)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print("✅ Saved", save_path)
    plt.close()


def show_same_subject_diff_poses(X_img, y, subject_id, n_poses=9, save_path=None):
    """Shows multiple images from same subject (first N samples) - treated as pose/variation."""
    idxs = np.where(y == subject_id)[0][:n_poses]
    fig, axes = plt.subplots(1, len(idxs), figsize=(2.5*len(idxs), 3.5))
    if len(idxs) == 1:
        axes = [axes]

    for ax, idx in zip(axes, idxs):
        ax.imshow(X_img[idx], cmap="gray")
        ax.axis("off")
        ax.set_title(f"Pose {idx%9 + 1}", fontsize=9, color='gray')

    plt.tight_layout()
    fig.suptitle(f"Subject {subject_id+1}: Pose Variations", fontsize=16, y=1.05)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print("✅ Saved", save_path)
    plt.close()



def reconstruct_image_with_different_k_eigenfaces(eigen_model, test_image, ks, H, W,
                                                  save_path="eigen_recon_k.png", show=True):
    """
    Reconstruct a single image using different k values.
    Prints MSE for each k and saves the plot.
    """
    if eigen_model.mean is None or eigen_model.W is None:
        raise ValueError("EigenFaces model must be fitted before calling this function.")

    x = test_image.reshape(1, -1).astype(np.float32)
    xc = x - eigen_model.mean

    n_plots = len(ks) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(3.5 * n_plots, 4))

    # Original
    axes[0].imshow(test_image, cmap="gray")
    axes[0].set_title("Original", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    print("\n" + "="*60)
    print("🧪 EigenFaces Reconstruction Quality vs k")
    print("="*60)

    for i, k in enumerate(ks):
        Wk = eigen_model.W[:, :k]
        coeff = xc @ Wk
        xr = eigen_model.mean + coeff @ Wk.T
        img_rec = xr.reshape(H, W)

        mse = np.mean((test_image.astype(np.float32) - img_rec.astype(np.float32)) ** 2)
        print(f"k={k:<5} | MSE={mse:.4f}")

        axes[i+1].imshow(img_rec, cmap="gray")
        axes[i+1].set_title(f"k={k}\nMSE={mse:.1f}", fontsize=11)
        axes[i+1].axis("off")

    fig.suptitle("EigenFaces: Reconstruction vs Components (k)", y=0.95, fontsize=16, fontweight="bold")
    plt.tight_layout()

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"\n✅ Saved reconstruction plot: {save_path}")

    if show:
        plt.show()
    plt.close()


def eigenfaces_accuracy_time_vs_k(eigen_model, X_train, y_train, X_test, y_test,
                                  ks, num_classes,
                                  save_prefix="eigen_k_analysis", show=True):
    """
    Runs EigenFaces for each k:
      - Prints accuracy and time in terminal
      - Saves 3 plots:
          1) Accuracy vs k
          2) Time vs k
          3) Dual-axis Accuracy & Time vs k
    """
    original_k = eigen_model.k
    accs = []
    times = []

    print("\n" + "="*70)
    print("📌 EigenFaces Accuracy & Time vs k")
    print("="*70)
    print(f"| {'k':^10} | {'Accuracy (%)':^15} | {'Time (s)':^15} |")
    print("-" * 50)

    for k in ks:
        eigen_model.k = int(k)

        t0 = time.time()
        eigen_model.fit(X_train, y_train, num_classes)
        preds = eigen_model.predict(X_test)
        t1 = time.time()

        acc = accuracy_score(y_test, preds) * 100
        duration = t1 - t0

        accs.append(acc)
        times.append(duration)

        print(f"| {k:^10} | {acc:^15.2f} | {duration:^15.4f} |")

    eigen_model.k = original_k

    # ===============================
    # Plot 1: Accuracy vs k
    # ===============================
    plt.figure(figsize=(8, 5))
    plt.plot(ks, accs, marker="o", linewidth=2.5)
    plt.xlabel("k (Eigenfaces Components)")
    plt.ylabel("Accuracy (%)")
    plt.title("EigenFaces: Accuracy vs k")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    out1 = f"{save_prefix}_accuracy.png"
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    print(f"\n✅ Saved: {out1}")
    if show:
        plt.show()
    plt.close()

    # ===============================
    # Plot 2: Time vs k
    # ===============================
    plt.figure(figsize=(8, 5))
    plt.plot(ks, times, marker="s", linewidth=2.5)
    plt.xlabel("k (Eigenfaces Components)")
    plt.ylabel("Time (seconds)")
    plt.title("EigenFaces: Time vs k")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    out2 = f"{save_prefix}_time.png"
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    print(f"✅ Saved: {out2}")
    if show:
        plt.show()
    plt.close()

    # ===============================
    # Plot 3: Dual-axis
    # ===============================
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.set_xlabel("k (Eigenfaces Components)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.plot(ks, accs, marker="o", linewidth=2.5, label="Accuracy")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Time (seconds)")
    ax2.plot(ks, times, marker="s", linewidth=2.5, linestyle="--", label="Time")

    plt.title("EigenFaces: Accuracy & Time Trade-off")
    fig.tight_layout()

    out3 = f"{save_prefix}_dual.png"
    plt.savefig(out3, dpi=200, bbox_inches="tight")
    print(f"✅ Saved: {out3}")

    if show:
        plt.show()
    plt.close()

    return accs, times


# =========================================================
# EXTRA: Compatible base + example saving for TensorFacesLiteRealLight
# =========================================================

def save_bases_eigen_vs_tensorlite(eigen, tensorlite, out="bases.png", H=96, W=96, n_show=8):
    """
    Saves EigenFaces components and TensorLite pixel bases (U_p columns) side-by-side.
    Works with TensorFacesLiteRealLight (has U_p).
    """
    if eigen.W is None:
        raise ValueError("EigenFaces model must be fitted first.")
    if tensorlite.U_p is None:
        raise ValueError("TensorFacesLite model must be fitted first.")

    plt.figure(figsize=(2*n_show, 4))

    # EigenFaces
    for i in range(n_show):
        ax = plt.subplot(2, n_show, i + 1)
        ef = eigen.W[:, i].reshape(H, W)
        ax.imshow(norm01(ef), cmap="gray")
        ax.axis("off")
        if i == 0:
            ax.set_title("EigenFaces (PCA)", loc="left", fontsize=11, fontweight="bold")

    # TensorFacesLite pixel basis
    for i in range(n_show):
        ax = plt.subplot(2, n_show, n_show + i + 1)
        tf = tensorlite.U_p[:, i].reshape(H, W)
        ax.imshow(norm01(tf), cmap="gray")
        ax.axis("off")
        if i == 0:
            ax.set_title("TensorFacesLite (U_p)", loc="left", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("✅ Saved", out)


def save_examples_eigen_vs_tensorlite(Xte, yte, preds, eigen, tensorlite, out="examples.png"):
    """
    Saves correct & wrong examples for TensorFacesLite predictions, with Eigen and TensorLite reconstructions.
    """
    idx_c = np.where(yte == preds)[0][:4]
    idx_w = np.where(yte != preds)[0][:4]
    idx = list(idx_c) + list(idx_w)
    n = len(idx)
    if n == 0:
        print("⚠️ No examples to save.")
        return

    plt.figure(figsize=(12, 3.2*n))

    for r, k in enumerate(idx):
        img, true_lbl, pred_lbl = Xte[k], yte[k], preds[k]
        is_correct = (true_lbl == pred_lbl)
        box_color = "#2ecc71" if is_correct else "#e74c3c"
        tag = "CORRECT" if is_correct else "WRONG"

        # Original
        ax = plt.subplot(n, 3, 3*r + 1)
        ax.imshow(img, cmap="gray")
        for spine in ax.spines.values():
            spine.set_color(box_color)
            spine.set_linewidth(3)
        ax.set_title(f"{tag}\nTrue: {true_lbl+1} | Pred: {pred_lbl+1}", color=box_color, fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

        # EigenFaces recon
        ax = plt.subplot(n, 3, 3*r + 2)
        ax.imshow(norm01(eigen.reconstruct(img)), cmap="gray")
        ax.set_title("EigenFaces Reconstruction", fontsize=10)
        ax.axis("off")

        # TensorFacesLite recon
        ax = plt.subplot(n, 3, 3*r + 3)
        ax.imshow(norm01(tensorlite.reconstruct(img)), cmap="gray")
        ax.set_title("TensorFacesLite Reconstruction", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print("✅ Saved", out)
    plt.close()

# =========================================================
# 7. MAIN EXECUTION
# =========================================================

def main():
    data_dir = "CroppedYaleB"

    # 1. LOAD
    images, labels, filenames = load_dataset(
        data_dir,
        img_size=(96, 96),
        max_per_class=64,  # Keeping it lighter for analysis speed
        max_classes=None
    )
    if len(images) == 0:
        raise RuntimeError("Dataset is empty!")

    # 2. SPLIT (Include filenames)
    X_train, X_test, y_train, y_test, fn_train, fn_test = train_test_split(
        images, labels, filenames,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    print(f"📊 Train: {len(X_train)} | Test: {len(X_test)} | Classes: {len(np.unique(labels))}")

    # 3. BUILD TENSOR (Train Data Only)
    D_train, subjects, lights, (H, W) = build_subject_light_tensor(X_train, y_train, fn_train)
    print(f"📦 Tensor Built: {D_train.shape} (Subjects, Lights, Pixels)")

    # --- EDA / Dataset Overview ---
    show_diff_subjects(images, labels, save_path="diff_subjects.png")
    show_same_subject_diff_illumination(images, labels, subject_id=0, n_lights=16, save_path="same_subject_light.png")
    show_same_subject_diff_poses(images, labels, subject_id=0, save_path="same_subject_pose.png")


    results_acc = {}
    results_time = {}

    # --- MODEL 1: EigenFaces ---
    print("\n🔄 Running EigenFaces...")
    start = time.time()
    eigen = EigenFaces(100)
    eigen.fit(X_train, y_train, len(subjects))
    preds_eigen = eigen.predict(X_test)
    t_eigen = time.time() - start
    acc_eigen = accuracy_score(y_test, preds_eigen) * 100
    results_acc["EigenFaces"] = acc_eigen
    results_time["EigenFaces"] = t_eigen
    print(f"   Acc: {acc_eigen:.2f}% | Time: {t_eigen:.2f}s")

    # --- MODEL 2: TensorFaces (Standard Run) ---
    print("\n🔄 Running TensorFaces...")
    start = time.time()
    tensor_model = TensorFacesLiteRealLight(H, W, ranks=(28, 10, 150), lam=0.01)
    tensor_model.fit(D_train, subjects, lights)
    preds_tensor = tensor_model.predict(X_test)
    t_tensor = time.time() - start
    acc_tensor = accuracy_score(y_test, preds_tensor) * 100
    results_acc["TensorFaces"] = acc_tensor
    results_time["TensorFaces"] = t_tensor
    print(f"   Acc: {acc_tensor:.2f}% | Time: {t_tensor:.2f}s")

     # --- MODEL 3: LBP ---
    print("\n🔄 Running LBP...")
    start = time.time()
    lbp = LBPClassifier()
    lbp.fit(X_train, y_train, len(subjects))
    preds_lbp = lbp.predict(X_test)
    t_lbp = time.time() - start
    acc_lbp = accuracy_score(y_test, preds_lbp) * 100
    results_acc["LBP"] = acc_lbp
    results_time["LBP"] = t_lbp
    print(f"   Acc: {acc_lbp:.2f}% | Time: {t_lbp:.2f}s")

    # --- MODEL 4: FisherFaces ---
    print("\n🔄 Running FisherFaces...")
    start = time.time()
    fisher = FisherFaces(pca_components=50)
    fisher.fit(X_train, y_train)
    preds_fisher = fisher.predict(X_test)
    t_fisher = time.time() - start
    acc_fisher = accuracy_score(y_test, preds_fisher) * 100
    results_acc["FisherFaces"] = acc_fisher
    results_time["FisherFaces"] = t_fisher
    print(f"   Acc: {acc_fisher:.2f}% | Time: {t_fisher:.2f}s")


    # --- VISUALIZATION: Comparison Plots ---
    print("\n📊 Generating Comparison Plots...")
    plot_accuracy_comparison(results_acc, "comparison_accuracy.png")
    plot_time_comparison(results_time, "comparison_time.png")

    # Extract and save EigenFaces bases
    extract_and_save_eigenfaces_bases(
        eigen_model=eigen,
        H=H, W=W,
        n_show=5,
        save_img="eigenfaces_bases.png",
        save_npy="eigenfaces_bases.npy"
    )

    # Extract and save TensorFaces bases
    extract_and_save_tensorfaces_bases(
        tensor_model=tensor_model,
        H=H, W=W,
        n_show=5,
        save_img="tensorfaces_bases.png",
        save_npy="tensorfaces_bases.npy"
    )


    reconstruct_image_with_different_k_eigenfaces(
    eigen_model=eigen,
    test_image=X_test[0],
    ks=[10, 30, 60, 100],
    H=H, W=W,
    save_path="analysis_eigen_recon.png"
)


    accs, times = eigenfaces_accuracy_time_vs_k(
    eigen, X_train, y_train, X_test, y_test,
    ks=[10, 30, 50, 100, 150, 300],
    num_classes=len(subjects),
    save_prefix="analysis_eigen"
)


    save_bases_eigen_vs_tensorlite(eigen, tensor_model, out="bases_eigen_tensorlite.png", H=H, W=W)
    save_examples_eigen_vs_tensorlite(X_test, y_test, preds_tensor, eigen, tensor_model, out="examples_eigen_tensorlite.png")


    # --- VISUALIZATION: Reconstructions ---
    print("\n🎨 Saving Reconstruction Examples...")
    # EigenFaces Recon
    save_single_reconstruction(X_test, eigen.reconstruct, title="EigenFaces Recon", out="recon_eigen.png")
    # TensorFaces Recon
    save_single_reconstruction(X_test, tensor_model.reconstruct, title="TensorFaces Recon", out="recon_tensor.png")

        # --- Detailed Visualizations ---
    show_and_save_predictions_with_reconstruction(
        X_test, y_test, preds_eigen,
        X_train, y_train,
        reconstruct_fn=eigen.reconstruct,
        method_name="EigenFaces",
        max_show=6,
        save_path="eigen_examples.png",
        layout="input_pred_recon"
    )

    show_and_save_predictions_with_reconstruction(
        X_test, y_test, preds_tensor,
        X_train, y_train,
        reconstruct_fn=tensor_model.reconstruct,
        method_name="TensorFaces (HOSVD)",
        max_show=6,
        save_path="tensor_examples.png",
        layout="input_pred_recon"
    )

    '''

    print("\n🎨 Generating 3-Column Visualization for EigenFaces...")
    show_predictions_with_recon_eigenfaces(
        X_test, y_test, preds_eigen,
        X_train, y_train,
        eigen_model=eigen,
        method_name="EigenFaces",
        max_show=6,
        save_path="examples_eigenfaces.png"
    )

    
    print("\n🎨 Generating 3-Column Visualization for TensorFaces...")
    show_predictions_with_recon(
        X_test, y_test, preds_tensor, 
        X_train, y_train, 
        reconstruct_fn=tensor_model.reconstruct,
        method_name="TensorFaces", 
        max_show=6,
        save_path="examples_tensorfaces.png"
    )
    '''


    #tensorfaces_hyperparams_analysis(D_train, subjects, lights, X_test, y_test, H, W)

    tensorfaces_single_param_analysis(
    D_train, subjects, lights,
    X_test, y_test, H, W,
    param_name="rL",
    param_values=[2, 5, 8, 10],
    fixed_rS=30,
    fixed_rL=10,
    fixed_rP=150,
    fixed_lam=0.01,
    save_prefix="analysis_tensor",
    show=True
)

    tensorfaces_single_param_analysis(
    D_train, subjects, lights,
    X_test, y_test, H, W,
    param_name="rP",
    param_values=[50, 100, 150, 200, 300],
    fixed_rS=30, fixed_rL=10, fixed_rP=150, fixed_lam=0.01,
    save_prefix="analysis_tensor"
)


    tensorfaces_single_param_analysis(
    D_train, subjects, lights,
    X_test, y_test, H, W,
    param_name="rS",
    param_values=[5, 10, 15, 20, 25, 30],
    fixed_rS=30, fixed_rL=10, fixed_rP=150, fixed_lam=0.01,
    save_prefix="analysis_tensor"
)



if __name__ == "__main__":
    main()
