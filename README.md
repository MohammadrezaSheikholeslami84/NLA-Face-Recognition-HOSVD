
# Face Recognition Using HOSVD (Tensorfaces)

This repository contains the implementation and report for the **final project of the Numerical Linear Algebra course** at **Amirkabir University of Technology (Tehran Polytechnic)**.  
The project investigates **face recognition using Higher-Order Singular Value Decomposition (HOSVD)** and compares tensor-based methods with classical linear algebra approaches.


## 📌 Course Information
- **Course:** Numerical Linear Algebra  
- **Instructor:** Dr. Mehdi Dehghan  
- **University:** Amirkabir University of Technology  
- **Semester:** Winter 1404  


## 🎯 Project Objective
Classical face recognition methods (such as Eigenfaces) vectorize images and apply matrix-based techniques like SVD or PCA.  
However, this process **destroys the inherent multi-dimensional structure** of image data.

This project aims to:
- Preserve the **multi-way structure** of face images using tensor representations
- Apply **HOSVD (Tucker Decomposition)** for dimensionality reduction
- Implement and evaluate **Tensorfaces**
- Compare tensor-based methods with classical techniques


## 🧠 Methods Implemented
The following face recognition approaches are implemented and evaluated:

### 1. Eigenfaces (PCA + SVD)
- Classical linear method
- Images vectorized into high-dimensional vectors
- Sensitive to illumination changes

### 2. Fisherfaces (PCA + LDA)
- Discriminative subspace learning
- Strong performance with lower computational cost

### 3. Local Binary Patterns (LBP)
- Texture-based local descriptor
- Robust to illumination variation

### 4. Tensorfaces (HOSVD-based)
- Uses **tensor modeling** of images
- Separates identity, illumination, and pixel structure
- Based on **Higher-Order Singular Value Decomposition (HOSVD)**


## 🧮 Mathematical Background
Key concepts covered in the report and code:
- Singular Value Decomposition (SVD)
- Tensor representation of image datasets
- Tensor unfolding (mode-n matricization)
- Mode-n tensor multiplication
- HOSVD and truncated HOSVD
- Multilinear dimensionality reduction


## 🗂 Dataset
- **Extended Yale B Face Dataset**
- 28 subjects
- 64 images per subject
- Total: **1792 images**
- Grayscale images resized to **96×96**

### Data Split
- **Training:** 80% (1433 images)
- **Testing:** 20% (359 images)
- Stratified split per subject


## ⚙️ Implementation Details
- Programming Language: **Python**
- Libraries:
  - NumPy, SciPy
  - scikit-learn
  - scikit-image
  - matplotlib, seaborn
  - PIL, tqdm

The code includes:
- Image preprocessing and normalization
- HOSVD implementation for 3D tensors
- Reconstruction of faces from reduced subspaces
- Classification and evaluation pipeline
- Visualization of eigenfaces and tensorfaces
- Hyperparameter sensitivity analysis


## 📊 Experimental Results

| Method        | Accuracy (%) | Time (s) |
|--------------|-------------|----------|
| Eigenfaces   | 42.62       | 16.19     |
| LBP          | 71.59       | 20.78    |
| Fisherfaces  | **85.79**   | **0.48** |
| Tensorfaces  | 72.70       | 40.66    |

### Observations
- **Fisherfaces** achieves the best accuracy with minimal runtime
- **Tensorfaces** shows strong robustness to illumination changes
- Tensor-based methods preserve structural information better
- HOSVD introduces higher computational cost compared to matrix methods


## 🔬 Hyperparameter Analysis
The project analyzes the effect of:
- Number of eigenfaces (k)
- Tensor ranks:  
  - Subject rank (rₛ)  
  - Illumination rank (rₗ)  
  - Pixel rank (rₚ)
- Regularization parameter (λ)

Trade-offs between **accuracy and computational cost** are thoroughly examined.


## 🖼 Visualizations
The implementation includes:
- Face reconstruction comparisons
- Correct vs incorrect predictions
- Eigenfaces and Tensorfaces basis visualization
- Accuracy and execution time plots


## 📁 Repository Structure

```
├── full-version-code.py          # Full implementation
├── Application_of_HOSVD_in_Image_Recognition.pdf

```

### 🧾 Report
A complete mathematical explanation and experimental analysis is provided in:

**`Application_of_HOSVD_in_Image_Recognition.pdf`**

The report covers:
- Theory of SVD and HOSVD
- Tensor algebra fundamentals
- Detailed derivation of Tensorfaces
- Experimental results and discussion

---

## 🚀 Conclusion
This project demonstrates how **multilinear algebra** and **tensor decompositions** provide a powerful framework for face recognition.  
While computationally more expensive, tensor-based methods offer better structural modeling and robustness compared to classical linear techniques.

