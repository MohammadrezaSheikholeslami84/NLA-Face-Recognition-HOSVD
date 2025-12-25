import os
import time
import math
import itertools
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

# Improved Style Configuration
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.titleweight'] = 'bold'

print("✅ Libraries imported successfully.")


# =========================================================
# 2. DATASET LOADER & PREPROCESSING
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


def load_dataset(data_dir, img_size=(96, 96), max_per_class=None, max_classes=None):
    """Loads images from directory structure: root/subject/image.pgm"""
    if not os.path.exists(data_dir):
        print(f"❌ Error: Dataset directory '{data_dir}' not found.")
        return np.array([]), np.array([])

    images, labels = [], []
    subjects = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    selected_subjects = subjects[:max_classes] if max_classes else subjects
    print(f"📂 Loading {len(selected_subjects)} classes...")

    for sid, sname in enumerate(selected_subjects):
        spath = os.path.join(data_dir, sname)
        files = [f for f in os.listdir(spath) if f.lower().endswith(('.pgm', '.jpg', '.png'))]
        
        current_files = files[:max_per_class] if max_per_class else files

        if sid % 5 == 0:
            print(f"  -> Loading subject {sid+1}/{len(selected_subjects)} ({len(current_files)} images)")

        for f in current_files:
            try:
                img = Image.open(os.path.join(spath, f)).convert('L')
                img = img.resize(img_size)
                images.append(preprocess_image(np.array(img)))
                labels.append(sid)
            except Exception as e:
                print(f"Skipped {f}: {e}")

    print(f"✅ Total Loaded: {len(images)} images from {len(np.unique(labels))} subjects.")
    return np.array(images), np.array(labels)


# =========================================================
# 3. METRICS & REPORTS
# =========================================================

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "precision_macro": precision_score(y_true, y_pred, average="macro") * 100,
        "precision_weighted": precision_score(y_true, y_pred, average="weighted") * 100,
        "recall_macro": recall_score(y_true, y_pred, average="macro") * 100,
        "recall_weighted": recall_score(y_true, y_pred, average="weighted") * 100,
        "f1_macro": f1_score(y_true, y_pred, average="macro") * 100,
        "f1_weighted": f1_score(y_true, y_pred, average="weighted") * 100,
    }


def print_full_report(y_true, y_pred, method_name="Method"):
    metrics = compute_metrics(y_true, y_pred)
    print("\n" + "="*70)
    print(f"📌 {method_name} - Detailed Metrics")
    print("="*70)
    for k, v in metrics.items():
        print(f"{k:<20} : {v:.2f}%")
    print("\n📄 Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    return metrics


# =========================================================
# 4. VISUALIZATION FUNCTIONS (ALL PRESERVED)
# =========================================================

def show_preprocessing_steps(original_img, mean_face, H, W, save_path=None, show=True):
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
    if show: plt.show()
    plt.close()


def show_diff_subjects(X_img, y, subject_ids=None, save_path=None, max_cols=7):
    if subject_ids is None: subject_ids = np.unique(y)
    n = len(subject_ids)
    n_cols = min(max_cols, n)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2.2 * n_rows), constrained_layout=True)
    if n == 1: axes = np.array([axes])
    axes = axes.flatten()

    for ax, sid in zip(axes, subject_ids):
        idx = np.where(y == sid)[0][0]
        ax.imshow(X_img[idx], cmap="gray")
        ax.set_title(f"Subject {sid + 1}", fontsize=10, fontweight='bold', color="#444")
        ax.axis("off")

    for ax in axes[n:]: ax.axis("off")
    fig.suptitle("Dataset Subjects Overview", fontsize=16, y=1.02)
    
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches="tight")
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
    for ax in axes[len(idxs):]: ax.axis("off")

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.suptitle(f"Subject {subject_id}: Illumination Variations", fontsize=16, y=0.95)
    
    if save_path: plt.savefig(save_path, dpi=200, bbox_inches="tight")
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
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix", out=None, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(np.float32)
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12) * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", cbar=True, square=True, linewidths=0.2)
    
    plt.title(title, fontsize=16, fontweight="bold", pad=15)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)

    if class_names is not None:
        plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=90, fontsize=8)
        plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0, fontsize=8)

    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=250, bbox_inches="tight")
        print("✅ Saved", out)
    plt.show()
    plt.close()


def plot_accuracy_comparison(acc_dict, out="accuracy_comparison.png"):
    methods = list(acc_dict.keys())
    accs = list(acc_dict.values())
    
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("viridis", n_colors=len(methods))
    bars = plt.bar(methods, accs, color=palette, edgecolor='black', alpha=0.85, width=0.6)

    plt.ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
    plt.title("Classification Accuracy by Method", fontsize=15, pad=20)
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, acc in zip(bars, accs):
        plt.text(
            bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{acc:.1f}%",
            ha="center", va="bottom", fontsize=12, fontweight='bold', color='#333',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
        )
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.show()
    print("✅ Saved", out)


def plot_time_comparison(time_dict, out="time_comparison.png"):
    methods = list(time_dict.keys())
    times = list(time_dict.values())

    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("Reds", n_colors=len(methods))[::-1]
    bars = plt.bar(methods, times, color=palette, edgecolor='black', alpha=0.85, width=0.6)

    plt.ylabel("Execution Time (seconds)", fontsize=12, fontweight='bold')
    plt.title("Computational Efficiency Comparison", fontsize=15, pad=20)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, t in zip(bars, times):
        plt.text(
            bar.get_x() + bar.get_width()/2, bar.get_height() + (max(times)*0.02),
            f"{t:.2f}s", ha="center", va="bottom", fontsize=11, color='#333'
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
    if save_path: plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show: plt.show()
    plt.close()


def show_and_save_predictions_with_reconstruction(
    X_test, y_test, y_pred, X_train, y_train, reconstruct_fn,
    method_name="Method", max_show=6, save_path=None, show=True
):
    idx_correct = np.where(y_test == y_pred)[0]
    idx_wrong   = np.where(y_test != y_pred)[0]
    n_correct = min(len(idx_correct), max_show // 2)
    n_wrong = min(len(idx_wrong), max_show - n_correct)
    
    selected_indices = np.concatenate([idx_correct[:n_correct], idx_wrong[:n_wrong]]).astype(int)
    n = len(selected_indices)
    if n == 0: return

    fig = plt.figure(figsize=(12, 3.8 * n))
    
    # Pre-fetch representatives
    class_representatives = {}
    for cls in np.unique(y_pred[selected_indices]):
        idx_cls = np.where(y_train == cls)[0]
        class_representatives[cls] = X_train[idx_cls[0]] if len(idx_cls) > 0 else np.zeros_like(X_test[0])

    for r, i in enumerate(selected_indices):
        img = X_test[i]
        true_lbl = y_test[i]
        pred_lbl = y_pred[i]
        is_correct = (true_lbl == pred_lbl)
        color = "#27ae60" if is_correct else "#c0392b"
        status_text = "✓ MATCH" if is_correct else "✗ MISMATCH"
        
        # Col 1: Input
        ax1 = fig.add_subplot(n, 3, 3*r + 1)
        ax1.imshow(img, cmap="gray")
        for spine in ax1.spines.values():
            spine.set_color(color)
            spine.set_linewidth(3)
        ax1.set_title(f"Test Input\n(True Label: S{true_lbl})", fontsize=10, fontweight='bold', pad=8)
        ax1.set_ylabel(status_text, fontsize=12, fontweight='bold', color=color, labelpad=10)
        ax1.set_xticks([]); ax1.set_yticks([])

        # Col 2: Prediction
        ax2 = fig.add_subplot(n, 3, 3*r + 2)
        rep_img = class_representatives.get(pred_lbl)
        ax2.imshow(rep_img, cmap="gray")
        ax2.set_title(f"Predicted Class Sample\n(Pred Label: S{pred_lbl})", fontsize=10, fontweight='bold', color="#444", pad=8)
        if not is_correct:
            ax2.text(5, 10, "?", color="white", fontsize=20, fontweight='bold',
                     bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', boxstyle='circle'))
        ax2.axis("off")

        # Col 3: Recon
        ax3 = fig.add_subplot(n, 3, 3*r + 3)
        rec = reconstruct_fn(img)
        ax3.imshow(rec, cmap="gray")
        ax3.set_title(f"{method_name}\nReconstruction", fontsize=10, color='#555', pad=8)
        ax3.axis("off")

    plt.suptitle(f"{method_name} Error Analysis", fontsize=16, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print("✅ Saved", save_path)
    if show: plt.show()
    plt.close()


def eigenfaces_accuracy_vs_k(eigen_model, X_train, y_train, X_test, y_test, ks, num_classes, save_path=None, show=True):
    accs = []
    original_k = eigen_model.k

    for k in ks:
        eigen_model.k = int(k)
        eigen_model.fit(X_train, y_train, num_classes)
        preds = eigen_model.predict(X_test)
        acc = (preds == y_test).mean() * 100
        accs.append(acc)

    eigen_model.k = original_k
    accs = np.array(accs)

    plt.figure(figsize=(8, 5))
    plt.plot(ks, accs, marker="o", markersize=8, linewidth=2.5, color="#1f77b4", label="Accuracy")
    plt.fill_between(ks, accs, color="#1f77b4", alpha=0.1)
    
    plt.xlabel("Number of Eigenfaces (k)", fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.title("Performance Scaling: EigenFaces", fontsize=14, pad=15)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    max_idx = np.argmax(accs)
    plt.annotate(f"Max: {accs[max_idx]:.1f}% @ k={ks[max_idx]}",
                 xy=(ks[max_idx], accs[max_idx]), xytext=(ks[max_idx], accs[max_idx] - 5),
                 arrowprops=dict(arrowstyle="->", lw=1), fontsize=11)

    plt.legend()
    if save_path: plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show: plt.show()
    plt.close()
    return accs.tolist()


def eigenfaces_time_vs_k(eigen_model, X_train, y_train, X_test, y_test, ks, num_classes, save_path=None, show=True):
    times = []
    original_k = eigen_model.k

    for k in ks:
        eigen_model.k = int(k)
        start = time.time()
        eigen_model.fit(X_train, y_train, num_classes)
        _ = eigen_model.predict(X_test)
        times.append(time.time() - start)

    eigen_model.k = original_k
    times = np.array(times)

    plt.figure(figsize=(8, 5))
    plt.plot(ks, times, marker="s", markersize=8, linewidth=2.5, color="#d62728", label="Fit+Predict Time")
    plt.xlabel("Number of Eigenfaces (k)", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.title("Time Complexity: EigenFaces", fontsize=14, pad=15)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    if save_path: plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show: plt.show()
    plt.close()
    return times.tolist()


def tensorfaces_single_param_analysis(X_train, y_train, X_test, y_test, H, W, param_name, param_values, fixed_r1, fixed_r2, fixed_r3, metric="cosine", save_prefix=None):
    accs, times = [], []

    for v in param_values:
        if param_name == "r1":   ranks = (v, fixed_r2, fixed_r3)
        elif param_name == "r2": ranks = (fixed_r1, v, fixed_r3)
        elif param_name == "r3": ranks = (fixed_r1, fixed_r2, v)
        else: raise ValueError("param_name must be r1, r2, or r3")

        start = time.time()
        model = TensorFacesHOSVD(H=H, W=W, num_classes=len(np.unique(y_train)), ranks=ranks, metric=metric)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        times.append(time.time() - start)
        accs.append(accuracy_score(y_test, preds) * 100)

    # Plot Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(param_values, accs, marker="o", markersize=8, linewidth=2.5, color="#2c3e50")
    plt.fill_between(param_values, accs, color="#2c3e50", alpha=0.05)
    plt.xlabel(f"Rank Parameter: {param_name}", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title(f"TensorFaces Sensitivity: {param_name}", fontsize=14, pad=15)
    plt.grid(True, linestyle="--", alpha=0.6)
    sns.despine()
    if save_prefix: plt.savefig(f"{save_prefix}_acc.png", dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()

    # Plot Time
    plt.figure(figsize=(8, 5))
    plt.plot(param_values, times, marker="D", markersize=8, linewidth=2.5, color="#e67e22")
    plt.xlabel(f"Rank Parameter: {param_name}", fontsize=12)
    plt.ylabel("Training + Test Time (s)", fontsize=12)
    plt.title(f"TensorFaces Computational Cost: {param_name}", fontsize=14, pad=15)
    plt.grid(True, linestyle="--", alpha=0.6)
    sns.despine()
    if save_prefix: plt.savefig(f"{save_prefix}_time.png", dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()

    return accs, times


def save_single_reconstruction(X, reconstruct_fn, idx=None, title="Face Reconstruction", out="reconstruction.png", show=True):
    if idx is None: idx = np.random.randint(0, len(X))
    img = X[idx]
    rec = reconstruct_fn(img)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(norm01(rec), cmap="gray")
    axes[1].set_title("Reconstruction", fontsize=12)
    axes[1].axis("off")

    plt.suptitle(f"{title} (index = {idx})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print("✅ Saved", out)
    if show: plt.show()
    plt.close()


def save_bases(eigen, tensor, out="bases.png", H=96, W=96, n_show=8):
    plt.figure(figsize=(2*n_show, 4))
    
    # EigenFaces
    for i in range(n_show):
        ax = plt.subplot(2, n_show, i + 1)
        ef = eigen.W[:, i].reshape(H, W)
        ax.imshow(norm01(ef), cmap="gray")
        ax.axis("off")
        if i == 0: ax.set_title("EigenFaces (PCA)", loc="left", fontsize=11, fontweight="bold")

    # TensorFaces
    for i in range(n_show):
        ax = plt.subplot(2, n_show, n_show + i + 1)
        Gk = tensor.G[:, :, i]                 
        tf = tensor.Uh @ Gk @ tensor.Uw.T      
        ax.imshow(norm01(tf), cmap="gray")
        ax.axis("off")
        if i == 0: ax.set_title("TensorFaces (HOSVD)", loc="left", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("✅ Saved", out)


def save_examples(Xte, yte, preds, eigen, tensor, out="examples.png"):
    idx_c = np.where(yte == preds)[0][:4]
    idx_w = np.where(yte != preds)[0][:4]
    idx = list(idx_c) + list(idx_w)
    n = len(idx)
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
        ax.set_title(f"{tag}\nTrue: {true_lbl} | Pred: {pred_lbl}", color=box_color, fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

        # EigenFaces
        ax = plt.subplot(n, 3, 3*r + 2)
        ax.imshow(norm01(eigen.reconstruct(img)), cmap="gray")
        ax.set_title("EigenFaces Reconstruction", fontsize=10)
        ax.axis("off")

        # TensorFaces
        ax = plt.subplot(n, 3, 3*r + 3)
        ax.imshow(norm01(tensor.reconstruct(img)), cmap="gray")
        ax.set_title("TensorFaces Reconstruction", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print("✅ Saved", out)


# =========================================================
# 5. MODEL CLASSES
# =========================================================

class EigenFaces:
    def __init__(self, n_components=100):
        self.k = n_components
        self.mean = None
        self.W = None
        self.class_means = None

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
        x = img.reshape(1, -1).astype(np.float32)
        z = (x - self.mean) @ self.W
        xr = self.mean + z @ self.W.T
        return xr.reshape(img.shape)


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
        self.pca = None
        self.lda = None

    def fit(self, images, labels):
        print("🔄 FisherFaces Training...")
        X = images.reshape(len(images), -1).astype(np.float32)
        self.pca = PCA(n_components=self.pca_components, whiten=True)
        Xp = self.pca.fit_transform(X)
        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(Xp, labels)

    def predict(self, images):
        X = images.reshape(len(images), -1).astype(np.float32)
        Xp = self.pca.transform(X)
        return self.lda.predict(Xp)


class TensorFacesHOSVD:
    def __init__(self, H=64, W=64, num_classes=28, ranks=(25, 25, 60), metric="cosine"):
        self.H, self.W = H, W
        self.C = num_classes
        self.r1, self.r2, self.r3 = ranks
        self.metric = metric
        self.Uh, self.Uw, self.Us, self.G = None, None, None, None
        self.knn = None

    def _unfold_mode1(self, X): return X.reshape(self.H, -1)
    def _unfold_mode2(self, X): return np.transpose(X, (1, 0, 2)).reshape(self.W, -1)
    def _unfold_mode3(self, X): return np.transpose(X, (2, 0, 1)).reshape(X.shape[2], -1)

    def _compute_core(self, X01):
        Y = np.tensordot(self.Uh.T, X01, axes=(1, 0))          
        Y = np.tensordot(self.Uw.T, Y, axes=(1, 1))            
        Y = np.transpose(Y, (1, 0, 2))                         
        return np.tensordot(Y, self.Us, axes=(2, 0))

    def fit(self, images, labels):
        print("🔄 TensorFaces Full HOSVD Training...")
        t0 = time.time()
        X01 = images.astype(np.float32) / 255.0               
        X01 = np.transpose(X01, (1, 2, 0))                    

        # HOSVD SVDs
        self.Uh, _, _ = np.linalg.svd(self._unfold_mode1(X01), full_matrices=False)
        self.Uh = self.Uh[:, :self.r1]
        
        self.Uw, _, _ = np.linalg.svd(self._unfold_mode2(X01), full_matrices=False)
        self.Uw = self.Uw[:, :self.r2]
        
        self.Us, _, _ = np.linalg.svd(self._unfold_mode3(X01), full_matrices=False)
        self.Us = self.Us[:, :self.r3]

        self.G = self._compute_core(X01)
        
        # Train kNN on core projections
        train_feats = [self.project_core(img).reshape(-1).astype(np.float32) for img in images]
        self.knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
        self.knn.fit(train_feats, labels)
        
        print(f"✅ TensorFaces HOSVD: Uh{self.Uh.shape}, Uw{self.Uw.shape}, Us{self.Us.shape}, G{self.G.shape} | Time: {time.time()-t0:.2f}s")

    def project_core(self, img):
        X = img.astype(np.float32) / 255.0
        return self.Uh.T @ X @ self.Uw

    def reconstruct(self, img):
        g = self.project_core(img)                                
        Xr = self.Uh @ g @ self.Uw.T                               
        return np.clip(Xr, 0, 1) * 255.0

    def predict(self, test_images):
        test_feats = [self.project_core(img).reshape(-1).astype(np.float32) 
                      for img in tqdm(test_images, desc="TensorFaces kNN Predict")]
        return self.knn.predict(np.array(test_feats))


# =========================================================
# 6. MAIN EXECUTION
# =========================================================

def main():
    data_dir = "CroppedYaleB"
    images, labels = load_dataset(data_dir, max_per_class=64, max_classes=None)
    if len(images) == 0: return

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    C = len(np.unique(labels))
    print(f"📊 Dataset Split -> Train: {len(X_train)} | Test: {len(X_test)} | Classes: {C}")

    # Visualization: Exploratory Data Analysis
    show_diff_subjects(images, labels, save_path="extyale_diff_subjects.png")
    show_same_subject_diff_illumination(images, labels, subject_id=1, n_lights=16, save_path="extyale_same_subject_diff_light.png")
    show_same_subject_diff_poses(images, labels, subject_id=0, save_path="extyale_same_subject_diff_pose.png")

    results_acc = {}
    results_time = {}

    # --- 1. EigenFaces ---
    start = time.time()
    eigen = EigenFaces(100)
    eigen.fit(X_train, y_train, C)
    eigen_preds = eigen.predict(X_test)
    eigen_time = time.time() - start
    eigen_acc = accuracy_score(y_test, eigen_preds) * 100
    
    print(f"EigenFaces acc: {eigen_acc:.2f}%")
    results_acc["EigenFaces"] = eigen_acc
    results_time["EigenFaces"] = eigen_time

    # Preprocessing visualization (using an example image)
    if len(X_test) > 10:
        show_preprocessing_steps(X_test[10], eigen.mean, X_test.shape[1], X_test.shape[2], save_path="preprocessing_steps.png")

    # --- 2. TensorFaces ---
    start = time.time()
    tensor = TensorFacesHOSVD(H=96, W=96, num_classes=C, ranks=(25, 25, 60), metric="cosine")
    tensor.fit(X_train, y_train)
    tensor_preds = tensor.predict(X_test)
    tensor_time = time.time() - start
    tensor_acc = accuracy_score(y_test, tensor_preds) * 100
    
    print(f"TensorFaces acc: {tensor_acc:.2f}%")
    results_acc["TensorFaces"] = tensor_acc
    results_time["TensorFaces"] = tensor_time

    # --- 3. LBP ---
    start = time.time()
    lbp = LBPClassifier(P=8, R=1, grid_x=8, grid_y=8)
    lbp.fit(X_train, y_train, C)
    lbp_preds = lbp.predict(X_test)
    lbp_time = time.time() - start
    lbp_acc = accuracy_score(y_test, lbp_preds) * 100
    
    results_acc["LBP"] = lbp_acc
    results_time["LBP"] = lbp_time

    # --- 4. FisherFaces ---
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

    # --- Reporting & Matrices ---
    print_full_report(y_test, eigen_preds, method_name="EigenFaces")
    plot_confusion_matrix(y_test, eigen_preds, title="EigenFaces Confusion Matrix (Norm %)", out="cm_eigenfaces_norm.png", normalize=True)

    print_full_report(y_test, tensor_preds, method_name="TensorFaces (HOSVD)")
    plot_confusion_matrix(y_test, tensor_preds, title="TensorFaces Confusion Matrix (Norm %)", out="cm_tensorfaces_norm.png", normalize=True)

    print_full_report(y_test, fisher_preds, method_name="FisherFaces")
    plot_confusion_matrix(y_test, fisher_preds, title="FisherFaces Confusion Matrix (Norm %)", out="cm_fisherfaces_norm.png", normalize=True)

    # --- Summary Table ---
    print("\n" + "="*60 + "\nFINAL COMPARISON\n" + "="*60)
    print(f"{'Method':<20} {'Accuracy (%)':<15} {'Time (s)':<10}")
    print("-"*60)
    for m in results_acc:
        print(f"{m:<20} {results_acc[m]:<15.2f} {results_time[m]:<10.2f}")

    # --- Detailed Visualizations ---
    show_and_save_predictions_with_reconstruction(X_test, y_test, eigen_preds, X_train, y_train, eigen.reconstruct, "EigenFaces", save_path="eigen_examples.png")
    show_and_save_predictions_with_reconstruction(X_test, y_test, tensor_preds, X_train, y_train, tensor.reconstruct, "TensorFaces (HOSVD)", save_path="tensor_examples.png")
    
    save_single_reconstruction(X_test, tensor.reconstruct, idx=10, title="TensorFaces Reconstruction", out="tensor_single_recon.png")
    save_bases(eigen, tensor)
    save_examples(X_test, y_test, tensor_preds, eigen, tensor)
    plot_accuracy_comparison(results_acc)
    plot_time_comparison(results_time)

    # --- Hyperparameter Analysis ---
    H, W = X_test.shape[1], X_test.shape[2]
    reconstruct_image_with_different_k_eigenfaces(eigen, X_test[0], ks=[10, 30, 60, 100], H=H, W=W, save_path="eigen_recon.png")

    ks_values = [10, 20, 40, 80, 120]
    eigenfaces_accuracy_vs_k(eigen, X_train, y_train, X_test, y_test, ks_values, num_classes=n_classes, save_path="eigen_acc_vs_k.png")
    eigenfaces_time_vs_k(eigen, X_train, y_train, X_test, y_test, ks_values, num_classes=n_classes, save_path="eigen_time_vs_k.png")

    # TensorFaces Param Sensitivity
    tensorfaces_single_param_analysis(X_train, y_train, X_test, y_test, H=96, W=96, param_name="r1", param_values=[10, 15, 20, 25, 30], fixed_r1=None, fixed_r2=35, fixed_r3=60, save_prefix="tensor_r1")
    tensorfaces_single_param_analysis(X_train, y_train, X_test, y_test, H=96, W=96, param_name="r2", param_values=[20, 30, 40, 50], fixed_r1=20, fixed_r2=None, fixed_r3=60, save_prefix="tensor_r2")
    tensorfaces_single_param_analysis(X_train, y_train, X_test, y_test, H=96, W=96, param_name="r3", param_values=[20, 40, 60, 80, 100], fixed_r1=20, fixed_r2=35, fixed_r3=None, save_prefix="tensor_r3")

if __name__ == "__main__":
    main()
