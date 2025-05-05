# Hierarchical Jigsaw Puzzle Solver (Gumbel-Sinkhorn Based)

This repository implements replication of results of 2nd experiment of the original paper (vanilla approach) as well as hierarchical approach to solving jigsaw puzzles using Gumbel-Sinkhorn networks on the MNIST dataset. The framework builds on the vanilla permutation learning architecture and extends it with a two-stage hierarchical ordering strategy for improved scalability and interpretability.

---

## Hierarchical Idea

Instead of solving a large jigsaw puzzle directly (which becomes combinatorially hard), we:
1. **Divide the image** into small patches (e.g., 64 patches of size 4×4 for MNIST 28×28 images).
2. **Group patches** using a deterministic rule (e.g., based on intensity or position).
3. **Stage 1: Intra-group Ordering**
   - Use a Gumbel-Sinkhorn-based model (`model_patch`) to learn the correct permutation of patches **within** each group.
4. **Stage 2: Inter-group Ordering**
   - Stack ordered patches from each group into a “super patch” and apply another Gumbel-Sinkhorn model (`model_group`) to learn the correct **group ordering**.
5. **Reconstruction**
   - Apply both predicted permutations to reconstruct the original image.

This hierarchical decomposition improves training efficiency and allows clear interpretability at multiple levels of the model.

---

## Directory Structure

```
hierarchical_jigsaw_puzzle/
│
├── data/                        # Stores image datasets and patch info
├── dataset_builder.py          # Utilities to build datasets with shuffled patches
├── explore.ipynb               # [Intuition Behind Hierarchical Grouping] Understanding MNIST dataset and grouping strategy
├── hierarchical_train.py       # Implements the hierarchical Gumbel-Sinkhorn architecture
├── model.py                    # Model definitions (shared by both vanilla and hierarchical)
├── puzzle_utils.py             # Helper functions for patching, reshaping, and reconstruction
├── README.md                   # You are here!
├── vanilla_train.py            # Vanilla permutation learning approach (original paper replication)
└── visualize.ipynb             # Visualizes grouping & ordering after hierarchical model training
```

---

## Usage

- Run `vanilla_train.py` to train a model directly on all patches (replicates results of the 2nd experiment of the original paper)
- Run `hierarchical_train.py` to train the two-stage hierarchical model.
- Use `visualize.ipynb` to inspect grouping and ordering outputs visually after training.

