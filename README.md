# ML4SCI Genie — GSoC 2026 Task Submission

**Author:** Anjali Roy 
**GitHub:** github.com/anjaliii255 | **Branch:** gsoc

---

## About

This repository contains solutions to the evaluation tasks for the **Genie project** under ML4SCI (Google Summer of Code 2026). The project focuses on deep learning for jet physics at the Large Hadron Collider - specifically, classifying quark-initiated vs. gluon-initiated jets using graph neural networks.

The dataset used is the CMS Quark-Gluon dataset (139,306 jet images of shape 125x125x3), where the three channels correspond to ECAL, HCAL, and charged particle track deposits.

---

## Tasks

### Common Task 1 : Convolutional Autoencoder

Trained a convolutional autoencoder to reconstruct jet images. The encoder compresses each 125x125x3 image through 4 strided convolution layers, and the decoder reconstructs it using transposed convolutions with a final bilinear interpolation to enforce exact output dimensions. The loss combines MSE with a sparsity penalty on empty calorimeter regions to prevent hallucination of spurious energy deposits.

| Metric | Result |
|--------|--------|
| SSIM | 0.9804 |
| MSE | 0.000197 |
| False Activation Rate | 0.33% |

### Common Task 2 : GNN-based Jet Classification

Each jet image is converted to a point cloud where every active pixel becomes a node with 7 features: normalised spatial coordinates (x, y), ECAL, HCAL, Tracks, total energy, and log energy. Seven global jet-level physics features are also extracted per jet. The full dataset of 139,306 jets was used for training.

Three architectures were explored before arriving at the final model. GAT converged slowly and peaked at AUC 0.7825. EdgeConv achieved 0.7999 peak but severely overfit. The final optimised pointwise GNN with global pooling, trained with cosine annealing and early stopping on the full dataset, achieved the best generalisation.

| Model | AUC |
|-------|-----|
| GAT | 0.7825 |
| EdgeConv (overfit) | 0.7999* |
| EdgeConv (final) | **0.8032** |

### Specific Task 4 : Non-local GNN

Implements a Hybrid EdgeConv + Multi-Head Non-Local GNN to capture long-range dependencies between jet constituents. The non-local block computes scaled dot-product attention across all node pairs (4 attention heads), and two attention stages are interleaved between EdgeConv layers, allowing the model to refine global context at two different feature scales.

| Model | AUC |
|-------|-----|
| Baseline GNN | 0.8032 |
| Hybrid EdgeConv + Multi-Head Non-Local GNN | **0.8050** |

The non-local GNN consistently outperforms the baseline throughout training, confirming that long-range constituent correlations carry discriminative information for quark-gluon separation.

---

## Repository Structure

```
Genie-Task/
├── common_task1_autoencoder/
│   ├── common_task1.ipynb
│   └── comparison.png
├── common_task2_gnn/
│   ├── common_task2.ipynb
│   └── Task2_result.png
├── specific_task/
│   ├── specific_task4.ipynb
│   └── specific_task4_results.png
└── data/
    └── README.md
```

---

## Setup

All notebooks were trained on Kaggle (NVIDIA T4 GPU). Dependencies: Python 3.10+, PyTorch 2.x, NumPy, h5py, scikit-learn, matplotlib, pytorch-msssim.

---

## References

- Wang et al. Non-local Neural Networks. CVPR 2018.
- Wang et al. Dynamic Graph CNN for Learning on Point Clouds. ACM TOG 2019.
- Komiske et al. Energy Flow Networks. JHEP 2019.
- CMS Collaboration. Quark-Gluon Jet Dataset. Zenodo.