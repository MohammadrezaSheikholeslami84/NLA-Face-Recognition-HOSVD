import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, time, math
import itertools
import pandas as pd
from tqdm import tqdm
import warnings
import seaborn as sns  # Added for better aesthetics

# Sklearn tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

# Image processing
from skimage.feature import local_binary_pattern

# Configuration
warnings.filterwarnings("ignore")

# --- IMPROVED STYLE CONFIGURATION ---
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.titleweight'] = 'bold'
np.random.seed(42)

print("✅ Libraries imported successfully.")

# =========================================================
# Dataset loader
# =========================================================
def preprocess_image(img):
    """
    Standardizes and normalizes an image array.
    """
    x = img.astype(np.float32)
    x = x - x.mean()
    x = x / (x.std() + 1e-6)
    # Scale to 0..255
    x = (x - x.min()) / (x.max() - x.min() + 1e-12)
    return (x * 255.0).astype(np.uint8)

def load_dataset(data_dir, img_size=(96,96), max_per_class=24, max_classes=28):
    """
    Loads images from directory structure: root/subject/image.pgm
    """
    if not os.path.exists(data_dir):
        print(f"❌ Error: Dataset directory '{data_dir}' not found.")
        return np.array([]), np.array([])

    images, labels = [], []
    subjects = sorted([d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d))])

    print(f"📂 Loading {min(max_classes,len(subjects))} classes, max {max_per_class} images/class...")

    for sid, sname in enumerate(subjects[:max_classes]):
        spath = os.path.join(data_dir, sname)
        files = [f for f in os.listdir(spath)
                 if f.lower().endswith(('.pgm','.jpg','.png'))]

        for f in files[:max_per_class]:
            try:
                img = Image.open(os.path.join(spath,f)).convert('L')
                img = img.resize(img_size)
                images.append(preprocess_image(np.array(img)))
                labels.append(sid)
            except Exception as e:
                print(f"Skipped {f}: {e}")

    return np.array(images), np.array(labels)


# =========================================================
# IMPROVED VISUALIZATION FUNCTIONS
# =========================================================

def show_preprocessing_steps(original_img, mean_face, H, W, save_path=None, show=True):
    if mean_face.ndim == 1:
        mean_img = mean_face.reshape(H, W)
    else:
        mean_img = mean_face

    # -------- compute centered image --------
    centered = original_img.astype(np.float32) - mean_img.astype(np.float32)
    centered_norm = centered - centered.min()
    centered_norm = centered_norm / (centered_norm.max() + 1e-12)

    # -------- plotting --------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Custom colormap for better contrast
    cmap = "gray"

    # Original
    axes[0].imshow(original_img, cmap=cmap)
    axes[0].set_title("Original Input", fontsize=14, pad=10)
    
    # Mean face
    axes[1].imshow(mean_img, cmap=cmap)
    axes[1].set_title("Global Mean Face", fontsize=14, pad=10)
    
    # Centered image
    axes[2].imshow(centered_norm, cmap=cmap)
    axes[2].set_title("Zero-Mean (Centered)", fontsize=14, pad=10)

    # Aesthetic Cleanup
    for ax in axes:
        ax.axis("off")
        # Add a subtle border
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
    if subject_ids is None:
        subject_ids = np.unique(y)

    n = len(subject_ids)
    n_cols = min(max_cols, n)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2 * n_cols, 2.2 * n_rows),
        constrained_layout=True
    )

    if n == 1: axes = np.array([axes])
    axes = axes.flatten()

    for ax, sid in zip(axes, subject_ids):
        idx = np.where(y == sid)[0][0]
        ax.imshow(X_img[idx], cmap="gray")
        ax.set_title(f"Subject {sid + 1}", fontsize=10, fontweight='bold', color="#444")
        ax.axis("off")

    # Turn off extra axes
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Dataset Subjects Overview", fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def show_same_subject_diff_illumination(X_img, y, subject_id, n_lights=16, save_path=None):
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
    fig.suptitle(f"Subject {subject_id}: Illumination Variations", fontsize=16, y=0.95)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def show_same_subject_diff_poses(X_img, y, subject_id, n_poses=9, save_path=None):
    idxs = np.where(y == subject_id)[0][:n_poses]
    
    fig, axes = plt.subplots(1, len(idxs), figsize=(2.5*len(idxs), 3.5))
    
    for ax, idx in zip(axes, idxs):
        ax.imshow(X_img[idx], cmap="gray")
        ax.axis("off")
        ax.set_title(f"Pose {idx%9 + 1}", fontsize=9, color='gray')

    plt.tight_layout()
    fig.suptitle(f"Subject {subject_id}: Pose Variations", fontsize=16, y=1.05)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_accuracy_comparison(acc_dict, out="accuracy_comparison.png"):
    methods = list(acc_dict.keys())
    accs = list(acc_dict.values())

    plt.figure(figsize=(10, 6))
    
    # Use a professional palette
    palette = sns.color_palette("viridis", n_colors=len(methods))
    
    bars = plt.bar(methods, accs, color=palette, edgecolor='black', alpha=0.85, width=0.6)

    plt.ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
    plt.title("Classification Accuracy by Method", fontsize=15, pad=20)
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Aesthetic labels on bars
    for bar, acc in zip(bars, accs):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 2,
            f"{acc:.1f}%",
            ha="center", va="bottom",
            fontsize=12, fontweight='bold', color='#333',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
        )
    
    # Remove top/right spines
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.show()
    print("✅ Saved", out)


def plot_time_comparison(time_dict, out="time_comparison.png"):
    methods = list(time_dict.keys())
    times = list(time_dict.values())

    plt.figure(figsize=(10, 6))
    
    # Use a warm palette for time (implies cost)
    palette = sns.color_palette("Reds", n_colors=len(methods))
    # Reverse so darker red = more time
    palette = palette[::-1]

    bars = plt.bar(methods, times, color=palette, edgecolor='black', alpha=0.85, width=0.6)

    plt.ylabel("Execution Time (seconds)", fontsize=12, fontweight='bold')
    plt.title("Computational Efficiency Comparison", fontsize=15, pad=20)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, t in zip(bars, times):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + (max(times)*0.02),
            f"{t:.2f}s",
            ha="center", va="bottom",
            fontsize=11, color='#333'
        )

    sns.despine()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.show()
    print("✅ Saved", out)


def reconstruct_image_with_different_k_eigenfaces(eigen_model, test_image, ks, H, W, save_path=None, show=True):
    x = test_image.reshape(1, -1).astype(np.float32)
    xc = x - eigen_model.mean     

    n_plots = len(ks) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(3.5 * n_plots, 4))
    
    # Original
    axes[0].imshow(test_image, cmap="gray")
    axes[0].set_title("Original", fontsize=13, fontweight='bold')
    axes[0].axis("off")

    for i, k in enumerate(ks):
        Wk = eigen_model.W[:, :k]  
        coeff = xc @ Wk
        xr = eigen_model.mean + coeff @ Wk.T
        img_rec = xr.reshape(H, W)

        axes[i+1].imshow(img_rec, cmap="gray")
        axes[i+1].set_title(f"k = {k}", fontsize=13)
        axes[i+1].axis("off")

    fig.suptitle("EigenFaces: Reconstruction Quality vs Components", y=0.95, fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def reconstructed_img_eigenfaces(img, mean_face, eigenfaces, k):
    x = img.reshape(1, -1).astype(np.float32)
    xc = x - mean_face
    Wk = eigenfaces[:, :k]
    xr = mean_face + (xc @ Wk) @ Wk.T
    return xr.reshape(img.shape)


def show_and_save_predictions_with_reconstruction(
    X_test, y_test, y_pred,
    X_train, y_train,       # Added: need training data to show predicted class example
    reconstruct_fn,
    method_name="Method",
    max_show=6,
    save_path=None,
    show=True
):
    """
    Shows: 
    1. Input Test Image
    2. An example image from the Predicted Class (to see why it might have confused them)
    3. The Reconstructed Image by the model
    """
    
    # Select samples (mix of correct and wrong)
    idx_correct = np.where(y_test == y_pred)[0]
    idx_wrong   = np.where(y_test != y_pred)[0]
    
    # Take up to max_show/2 from each, ensuring we don't exceed bounds
    n_correct = min(len(idx_correct), max_show // 2)
    n_wrong = min(len(idx_wrong), max_show - n_correct)
    
    selected_indices = np.concatenate([idx_correct[:n_correct], idx_wrong[:n_wrong]]).astype(int)
    n = len(selected_indices)
    
    if n == 0: return

    # Create figure with 3 columns
    fig = plt.figure(figsize=(12, 3.8 * n))
    
    # Pre-calculate representative images for each class (mean or first sample)
    # Using the first sample found in X_train for simplicity and clarity
    class_representatives = {}
    unique_preds = np.unique(y_pred[selected_indices])
    for cls in unique_preds:
        # Find first image of this class in training set
        idx_cls = np.where(y_train == cls)[0]
        if len(idx_cls) > 0:
            class_representatives[cls] = X_train[idx_cls[0]]
        else:
            class_representatives[cls] = np.zeros_like(X_test[0]) # Fallback

    for r, i in enumerate(selected_indices):
        img = X_test[i]
        true_lbl = y_test[i]
        pred_lbl = y_pred[i]
        is_correct = (true_lbl == pred_lbl)

        # Colors
        color = "#27ae60" if is_correct else "#c0392b"  # Green / Red
        status_text = "✓ MATCH" if is_correct else "✗ MISMATCH"
        
        # --- Column 1: Input Test Image ---
        ax1 = fig.add_subplot(n, 3, 3*r + 1)
        ax1.imshow(img, cmap="gray")
        for spine in ax1.spines.values():
            spine.set_color(color)
            spine.set_linewidth(3)
        
        ax1.set_title(f"Test Input\n(True Label: S{true_lbl})", fontsize=10, fontweight='bold', pad=8)
        ax1.set_ylabel(status_text, fontsize=12, fontweight='bold', color=color, labelpad=10)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # --- Column 2: Predicted Class Example ---
        ax2 = fig.add_subplot(n, 3, 3*r + 2)
        rep_img = class_representatives.get(pred_lbl)
        
        ax2.imshow(rep_img, cmap="gray")
        ax2.set_title(f"Predicted Class Sample\n(Pred Label: S{pred_lbl})", fontsize=10, fontweight='bold', color="#444", pad=8)
        
        # Add an arrow or indicator if wrong
        if not is_correct:
            ax2.text(5, 10, "?", color="white", fontsize=20, fontweight='bold',
                     bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', boxstyle='circle'))
            
        for spine in ax2.spines.values():
            spine.set_visible(True)
            spine.set_color('#666')
            spine.set_linestyle('--')
            
        ax2.axis("off")

        # --- Column 3: Reconstruction ---
        ax3 = fig.add_subplot(n, 3, 3*r + 3)
        rec = reconstruct_fn(img)
        ax3.imshow(rec, cmap="gray")
        ax3.set_title(f"{method_name}\nReconstruction", fontsize=10, color='#555', pad=8)
        ax3.axis("off")

    plt.suptitle(f"{method_name} Error Analysis: Input vs Prediction vs Reconstruction", fontsize=16, y=1.01)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print("✅ Saved", save_path)

    if show:
        plt.show()
    plt.close()



def eigenfaces_accuracy_vs_k(eigen_model, X_train, y_train, X_test, y_test, ks, save_path=None, show=True):
    accs = []
    Xtr = X_train.reshape(len(X_train), -1) - eigen_model.mean
    Xte = X_test.reshape(len(X_test), -1) - eigen_model.mean

    for k in ks:
        Wk = eigen_model.W[:, :k]
        Ztr = Xtr @ Wk
        Zte = Xte @ Wk
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(Ztr, y_train)
        accs.append(knn.score(Zte, y_test) * 100)

    plt.figure(figsize=(8, 5))
    
    # Enhanced Line Plot
    plt.plot(ks, accs, marker="o", markersize=8, linewidth=2.5, color="#1f77b4", label="Accuracy")
    plt.fill_between(ks, accs, color="#1f77b4", alpha=0.1) # Shaded area
    
    plt.xlabel("Number of Eigenfaces (k)", fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.title("Performance Scaling: EigenFaces", fontsize=14, pad=15)
    
    plt.grid(True, linestyle="--", alpha=0.6)
    sns.despine()
    
    # Annotate max
    max_acc = max(accs)
    max_k = ks[np.argmax(accs)]
    plt.annotate(f'Max: {max_acc:.1f}%', xy=(max_k, max_acc), xytext=(max_k, max_acc-5),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=11)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return accs


def eigenfaces_time_vs_k(eigen_model, X_train, y_train, X_test, y_test, ks, save_path=None, show=True):
    times = []
    Xtr = X_train.reshape(len(X_train), -1) - eigen_model.mean
    Xte = X_test.reshape(len(X_test), -1) - eigen_model.mean

    for k in ks:
        start = time.time()
        Wk = eigen_model.W[:, :k]
        Ztr = Xtr @ Wk
        Zte = Xte @ Wk
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(Ztr, y_train)
        knn.predict(Zte)
        times.append(time.time() - start)

    plt.figure(figsize=(8, 5))
    plt.plot(ks, times, marker="s", markersize=8, linewidth=2.5, color="#d62728", label="Inference Time")
    
    plt.xlabel("Number of Eigenfaces (k)", fontsize=12)
    plt.ylabel("Inference Time (seconds)", fontsize=12)
    plt.title("Time Complexity: EigenFaces", fontsize=14, pad=15)
    plt.grid(True, linestyle="--", alpha=0.6)
    sns.despine()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return times


def tensorfaces_single_param_analysis(X_train, y_train, X_test, y_test, H, W, param_name, param_values, fixed_r1, fixed_r2, fixed_r3, metric="cosine", save_prefix=None):
    accs = []
    times = []

    for v in param_values:
        if param_name == "r1":   ranks = (v, fixed_r2, fixed_r3)
        elif param_name == "r2": ranks = (fixed_r1, v, fixed_r3)
        elif param_name == "r3": ranks = (fixed_r1, fixed_r2, v)
        else: raise ValueError("param_name must be r1, r2, or r3")

        start = time.time()
        model = TensorFacesHOSVD(H=H, W=W, num_classes=len(np.unique(y_train)), ranks=ranks, metric=metric)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        elapsed = time.time() - start

        accs.append(accuracy_score(y_test, preds) * 100)
        times.append(elapsed)

    # -------- Accuracy plot --------
    plt.figure(figsize=(8, 5))
    plt.plot(param_values, accs, marker="o", markersize=8, linewidth=2.5, color="#2c3e50")
    plt.fill_between(param_values, accs, color="#2c3e50", alpha=0.05)
    
    plt.xlabel(f"Rank Parameter: {param_name}", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title(f"TensorFaces Sensitivity: {param_name}", fontsize=14, pad=15)
    plt.grid(True, linestyle="--", alpha=0.6)
    sns.despine()

    if save_prefix:
        plt.savefig(f"{save_prefix}_acc.png", dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()

    # -------- Time plot --------
    plt.figure(figsize=(8, 5))
    plt.plot(param_values, times, marker="D", markersize=8, linewidth=2.5, color="#e67e22")
    
    plt.xlabel(f"Rank Parameter: {param_name}", fontsize=12)
    plt.ylabel("Training + Test Time (s)", fontsize=12)
    plt.title(f"TensorFaces Computational Cost: {param_name}", fontsize=14, pad=15)
    plt.grid(True, linestyle="--", alpha=0.6)
    sns.despine()

    if save_prefix:
        plt.savefig(f"{save_prefix}_time.png", dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()

    return accs, times

def save_single_reconstruction(
    X,
    reconstruct_fn,
    idx=None,
    title="Face Reconstruction",
    out="reconstruction.png",
    show=True
):
    """
    Shows and saves:
    [ Original Image | Reconstructed Image ]

    reconstruct_fn: function(img) -> reconstructed_img
    """

    # -------- select image --------
    if idx is None:
        idx = np.random.randint(0, len(X))

    img = X[idx]
    rec = reconstruct_fn(img)

    # -------- plotting --------
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(norm01(rec), cmap="gray")
    axes[1].set_title("Reconstruction", fontsize=12)
    axes[1].axis("off")

    plt.suptitle(
        f"{title} (index = {idx})",
        fontsize=14, y=1.02
    )

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print("✅ Saved", out)

    if show:
        plt.show()
    plt.close()

# =========================================================
# EigenFaces
# =========================================================

class EigenFaces:
    def __init__(self, n_components=100):
        self.k = n_components

    def fit(self, images, labels, num_classes):
        print("🔄 EigenFaces Training...")
        X = images.reshape(len(images), -1).astype(np.float32)
        self.mean = X.mean(axis=0)
        Xc = X - self.mean

        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.W = Vt.T[:, :self.k]

        Z = Xc @ self.W
        self.class_means = np.zeros((num_classes, self.k))
        for c in range(num_classes):
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
        x = img.reshape(1,-1).astype(np.float32)
        z = (x-self.mean)@self.W
        xr = self.mean + z@self.W.T
        return xr.reshape(img.shape)

class LBPClassifier:
    def __init__(self, P=8, R=1, grid_x=8, grid_y=8, num_bins=59):
        self.P = P
        self.R = R
        self.gx = grid_x
        self.gy = grid_y
        self.num_bins = num_bins

    def extract_lbp(self, img):
        lbp = local_binary_pattern(img, self.P, self.R, method="uniform")
        h, w = img.shape
        hx, wy = h // self.gx, w // self.gy

        features = []
        for i in range(self.gx):
            for j in range(self.gy):
                block = lbp[i*hx:(i+1)*hx, j*wy:(j+1)*wy]
                hist, _ = np.histogram(
                    block.ravel(),
                    bins=self.num_bins,
                    range=(0, self.num_bins),
                    density=True
                )
                features.append(hist)

        return np.concatenate(features)

    def fit(self, images, labels, num_classes):
        print("🔄 LBP Training...")
        feats = np.array([self.extract_lbp(img) for img in images])

        self.class_means = np.zeros((num_classes, feats.shape[1]))
        for c in range(num_classes):
            self.class_means[c] = feats[labels == c].mean(axis=0)

    def predict(self, images):
        preds = []
        for img in images:
            feat = self.extract_lbp(img)
            d = np.linalg.norm(self.class_means - feat, axis=1)
            preds.append(np.argmin(d))
        return np.array(preds)


class FisherFaces:
    def __init__(self, pca_components=100):
        self.pca_components = pca_components

    def fit(self, images, labels):
        print("🔄 FisherFaces Training...")
        X = images.reshape(len(images), -1).astype(np.float32)

        # PCA
        self.pca = PCA(n_components=self.pca_components, whiten=True)
        Xp = self.pca.fit_transform(X)

        # LDA
        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(Xp, labels)

    def predict(self, images):
        X = images.reshape(len(images), -1).astype(np.float32)
        Xp = self.pca.transform(X)
        return self.lda.predict(Xp)


# =========================================================
# TensorFaces (HOSVD + Core-space cosine classifier)
# =========================================================

class TensorFacesHOSVD:
    """
    Full HOSVD TensorFaces on a 3rd-order tensor.
    """

    def __init__(self, H=64, W=64, num_classes=28, ranks=(25, 25, 60), metric="cosine"):
        self.H, self.W = H, W
        self.C = num_classes
        self.r1, self.r2, self.r3 = ranks
        self.metric = metric

        self.Uh = None
        self.Uw = None
        self.Us = None
        self.G = None 

        self.class_means = None 
        self.class_norms = None 

        self.train_feats = None
        self.knn = None


    def _unfold_mode1(self, X):
        return X.reshape(self.H, -1)

    def _unfold_mode2(self, X):
        return np.transpose(X, (1, 0, 2)).reshape(self.W, -1)

    def _unfold_mode3(self, X):
        N = X.shape[2]
        return np.transpose(X, (2, 0, 1)).reshape(N, -1)

    def _compute_core(self, X01):
        Y = np.tensordot(self.Uh.T, X01, axes=(1, 0))          
        Y = np.tensordot(self.Uw.T, Y, axes=(1, 1))            
        Y = np.transpose(Y, (1, 0, 2))                         
        G = np.tensordot(Y, self.Us, axes=(2, 0))              
        return G

    def fit(self, images, labels):
        print("🔄 TensorFaces Full HOSVD Training...")
        t0 = time.time()

        N = len(images)
        assert images.shape[1] == self.H and images.shape[2] == self.W, \
            "Image size must match (H,W)."

        X01 = images.astype(np.float32) / 255.0               
        X01 = np.transpose(X01, (1, 2, 0))                    

        # ----- HOSVD: compute Uh, Uw, Us -----
        # Mode-1
        X1 = self._unfold_mode1(X01)                          
        U1, _, _ = np.linalg.svd(X1, full_matrices=False)
        self.Uh = U1[:, :self.r1]                             

        # Mode-2
        X2 = self._unfold_mode2(X01)                          
        U2, _, _ = np.linalg.svd(X2, full_matrices=False)
        self.Uw = U2[:, :self.r2]                             

        # Mode-3
        X3 = self._unfold_mode3(X01)                          
        U3, _, _ = np.linalg.svd(X3, full_matrices=False)
        self.Us = U3[:, :self.r3]                             

        # ----- Core tensor -----
        self.G = self._compute_core(X01)                      

        G_flat = self.G.reshape(self.r1 * self.r2, self.r3).T          
        S_train = self.Us                                              
        g_train_flat = S_train @ G_flat                                

        train_feats = []
        for img in images:
            g = self.project_core(img).reshape(-1).astype(np.float32)
            train_feats.append(g)

        self.train_feats = np.array(train_feats)

        # ---------- kNN classifier ----------
        self.knn = KNeighborsClassifier(
            n_neighbors=1,        # می‌تونی 3 هم تست کنی
            metric="euclidean"
        )
        self.knn.fit(self.train_feats, labels)

        print(f"✅ TensorFaces HOSVD: Uh{self.Uh.shape}, Uw{self.Uw.shape}, Us{self.Us.shape}, G{self.G.shape}"
              f" | Time: {time.time()-t0:.2f}s")

    # ---------- Projection / Reconstruction ----------
    def project_core(self, img):
        X = img.astype(np.float32) / 255.0
        g = self.Uh.T @ X @ self.Uw                               
        return g

    def reconstruct(self, img):
        g = self.project_core(img)                                
        Xr = self.Uh @ g @ self.Uw.T                               
        return np.clip(Xr, 0, 1) * 255.0

    # ---------- Prediction ----------
    def predict(self, test_images):
        test_feats = []

        for img in tqdm(test_images, desc="TensorFaces kNN Predict"):
            g = self.project_core(img).reshape(-1).astype(np.float32)
            test_feats.append(g)

        test_feats = np.array(test_feats)
        return self.knn.predict(test_feats)


# =========================================================
# Visualization helpers (Additional)
# =========================================================

def norm01(x):
    x=x.astype(np.float32)
    return (x-x.min())/(x.max()-x.min()+1e-12)

def save_bases(eigen, tensor, out="bases.png", H=96, W=96, n_show=8):
    plt.figure(figsize=(2*n_show, 4))

    # ================= EigenFaces =================
    for i in range(n_show):
        ax = plt.subplot(2, n_show, i + 1)
        ef = eigen.W[:, i].reshape(H, W)
        ax.imshow(norm01(ef), cmap="gray")
        ax.axis("off")
        if i == 0:
            ax.set_title("EigenFaces (PCA)", loc="left", fontsize=11, fontweight="bold")

    # ================= TensorFaces =================
    for i in range(n_show):
        ax = plt.subplot(2, n_show, n_show + i + 1)

        # --- REAL TensorFace from Core ---
        Gk = tensor.G[:, :, i]                 # (r1, r2)
        tf = tensor.Uh @ Gk @ tensor.Uw.T      # (H, W)

        ax.imshow(norm01(tf), cmap="gray")
        ax.axis("off")
        if i == 0:
            ax.set_title("TensorFaces (HOSVD)", loc="left", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("✅ Saved", out)

def save_examples(Xte, yte, preds, eigen, tensor, out="examples.png"):
    # pick 4 correct + 4 wrong
    idx_c = np.where(yte == preds)[0][:4]
    idx_w = np.where(yte != preds)[0][:4]
    idx = list(idx_c) + list(idx_w)

    n = len(idx)
    fig = plt.figure(figsize=(12, 3.2*n))

    for r, k in enumerate(idx):
        img = Xte[k]
        true_lbl = yte[k]
        pred_lbl = preds[k]
        is_correct = (true_lbl == pred_lbl)
        
        # Color coding
        box_color = "#2ecc71" if is_correct else "#e74c3c"
        tag = "CORRECT" if is_correct else "WRONG"

        # -------- Original --------
        ax = plt.subplot(n, 3, 3*r + 1)
        ax.imshow(img, cmap="gray")
        
        # Add colored border
        for spine in ax.spines.values():
            spine.set_color(box_color)
            spine.set_linewidth(3)
            
        ax.set_title(
            f"{tag}\nTrue: {true_lbl} | Pred: {pred_lbl}",
            color=box_color, fontsize=11, fontweight="bold"
        )
        ax.set_xticks([])
        ax.set_yticks([])

        # -------- EigenFaces --------
        ax = plt.subplot(n, 3, 3*r + 2)
        ax.imshow(norm01(eigen.reconstruct(img)), cmap="gray")
        ax.set_title("EigenFaces Reconstruction", fontsize=10)
        ax.axis("off")

        # -------- TensorFaces --------
        ax = plt.subplot(n, 3, 3*r + 3)
        ax.imshow(norm01(tensor.reconstruct(img)), cmap="gray")
        ax.set_title("TensorFaces Reconstruction", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print("✅ Saved", out)


# =========================================================
# Main
# =========================================================

def main():
    data_dir="CroppedYaleB"

    images,labels = load_dataset(data_dir)
    if len(images) == 0:
        return

    X_train,X_test,y_train,y_test = train_test_split(
        images,labels,test_size=0.3,stratify=labels,random_state=42)
    
    show_diff_subjects(
        images, labels,
        save_path="extyale_diff_subjects.png"
    )
    
    show_same_subject_diff_illumination(
        images, labels,
        subject_id=1,
        n_lights=16, 
        save_path="extyale_same_subject_diff_light.png"
    )
    
    show_same_subject_diff_poses(
        images, labels,
        subject_id=0,
        save_path="extyale_same_subject_diff_pose.png"
    )
    

    C=len(np.unique(labels))
    print(f"📊 Train:{len(X_train)} Test:{len(X_test)} Classes:{C}")

    results_acc = {}
    results_time = {}

    # --- EigenFaces ---
    start = time.time()
    eigen = EigenFaces(100)
    eigen.fit(X_train, y_train, C)
    eigen_preds = eigen.predict(X_test)
    eigen_time = time.time() - start
    

    eigen_acc = accuracy_score(y_test, eigen_preds) * 100
    print("EigenFaces acc:",eigen_acc)


    idx = 10
    if idx < len(X_test):
        original_img = X_test[idx]
        mean_face = eigen.mean    
        H, W = original_img.shape
        show_preprocessing_steps(
            original_img=original_img,
            mean_face=mean_face,
            H=H, W=W,
            save_path="preprocessing_steps.png"
        )

    results_acc["EigenFaces"] = eigen_acc
    results_time["EigenFaces"] = eigen_time

    # --- TensorFaces ---
    start = time.time()
    tensor = TensorFacesHOSVD(H=96, W=96, num_classes=C, ranks=(25,25,60), metric="cosine")
    tensor.fit(X_train, y_train)
    tensor_preds = tensor.predict(X_test)
    tensor_acc = accuracy_score(y_test, tensor_preds) * 100
    tensor_time = time.time() - start

    print("TensorFaces acc:", tensor_acc)

    results_acc["TensorFaces"] = tensor_acc
    results_time["TensorFaces"] = tensor_time

    # --- LBP ---
    start = time.time()
    lbp = LBPClassifier(P=8, R=1, grid_x=8, grid_y=8)
    lbp.fit(X_train, y_train, C)
    lbp_preds = lbp.predict(X_test)
    lbp_time = time.time() - start
    lbp_acc = accuracy_score(y_test, lbp_preds) * 100

    results_acc["LBP"] = lbp_acc
    results_time["LBP"] = lbp_time


    # --- FisherFaces ---
    start = time.time()
    n_classes = len(np.unique(labels))
    pca_dim = min(100, len(X_train) - n_classes)
    fisher = FisherFaces(pca_components=pca_dim)
    fisher.fit(X_train, y_train)
    fisher_preds = fisher.predict(X_test)
    fisher_time = time.time() - start
    fisher_acc = accuracy_score(y_test, fisher_preds) * 100

    results_acc["FisherFaces"] = fisher_acc
    results_time["FisherFaces"] = fisher_time

    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"{'Method':<20} {'Accuracy (%)':<15} {'Time (s)':<10}")
    print("-"*60)

    for m in results_acc:
        print(f"{m:<20} {results_acc[m]:<15.2f} {results_time[m]:<10.2f}")

    # --- Results Visualization ---
    show_and_save_predictions_with_reconstruction(
        X_test=X_test,
        y_test=y_test,
        y_pred=eigen_preds,
        X_train=X_train,   
        y_train=y_train,   
        reconstruct_fn=eigen.reconstruct,     
        method_name="EigenFaces",
        max_show=8,
        save_path="eigen_examples.png"
    )
    

    # برای TensorFaces
    show_and_save_predictions_with_reconstruction(
        X_test=X_test,
        y_test=y_test,
        y_pred=tensor_preds,
        X_train=X_train,   
        y_train=y_train,    
        reconstruct_fn=tensor.reconstruct,    
        method_name="TensorFaces (HOSVD)",
        max_show=8,
        save_path="tensor_examples.png"
    )

    save_single_reconstruction(
    X=X_test,
    reconstruct_fn=tensor.reconstruct,
    idx=10,
    title="TensorFaces Reconstruction",
    out="tensor_single_recon.png"
)

    save_bases(eigen,tensor)
    save_examples(X_test,y_test,tensor_preds,eigen,tensor)
    plot_accuracy_comparison(results_acc)
    plot_time_comparison(results_time)

    # --- Analysis ---
    H, W = X_test.shape[1], X_test.shape[2]
    reconstruct_image_with_different_k_eigenfaces(
        eigen,
        X_test[0],
        ks=[10, 30, 60, 100],
        H=H, W=W,
        save_path="eigen_recon.png"
    )


    ks = [10, 20, 40, 80, 120]

    eigenfaces_accuracy_vs_k(
        eigen,
        X_train, y_train,
        X_test, y_test,
        ks,
        save_path="eigen_acc_vs_k.png"
    )

    eigenfaces_time_vs_k(
        eigen,
        X_train, y_train,
        X_test, y_test,
        ks,
        save_path="eigen_time_vs_k.png"
    )

    tensorfaces_single_param_analysis(
        X_train, y_train,
        X_test, y_test,
        H=96, W=96,
        param_name="r1",
        param_values=[10, 15, 20, 25, 30],
        fixed_r1=None,          
        fixed_r2=35,
        fixed_r3=60,
        save_prefix="tensor_r1"
    )
    
    tensorfaces_single_param_analysis(
        X_train, y_train,
        X_test, y_test,
        H=96, W=96,
        param_name="r2",
        param_values=[20, 30, 40, 50],
        fixed_r1=20,
        fixed_r2=None,          
        fixed_r3=60,
        save_prefix="tensor_r2"
    )


    tensorfaces_single_param_analysis(
        X_train, y_train,
        X_test, y_test,
        H=96, W=96,
        param_name="r3",
        param_values=[20, 40, 60, 80, 100],
        fixed_r1=20,
        fixed_r2=35,
        fixed_r3=None,          
        save_prefix="tensor_r3"
    )
    

if __name__=="__main__":
    main()
