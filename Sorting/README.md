# Learning Latent Permutations with Gumbel-Sinkhorn

This repository contains implementations related to learning to sort sequences using neural networks and the Gumbel-Sinkhorn operator, based on the concepts presented in related research papers.

## Implementations

Two main approaches are implemented in separate directories:

1.  **`Sorting_Same_As_Paper/`:**
    *   **File:** `sort.py`
    *   **Description:** This implementation closely follows the original Gumbel-Sinkhorn paper's approach for learning sorting permutations. It uses a simple element-wise MLP to generate scores, applies Gumbel-Sinkhorn to get a soft permutation matrix during training, and uses the Hungarian algorithm (via `scipy.optimize.linear_sum_assignment`) for hard permutation inference during evaluation. It does **not** explicitly use context from the entire input sequence when scoring individual elements.

2.  **`Sorting_With_Context/`:**
    *   **File:** `sort.py`
    *   **Description:** This implementation extends the basic sorting network by incorporating **context** from the entire input sequence, inspired by the **Deep Sets** architecture.
        *   An MLP (`g` - layers `ec1`, `ec2`) embeds each input element individually.
        *   These embeddings are summed across the sequence to produce an aggregated representation (`s`).
        *   Another MLP (`f` - layers `ec1_`, `ec2_`) processes this sum to generate a fixed-size context vector (`c`).
        *   The final score generation for each element considers both its original value (processed by `fc1`) and the global context vector (`c`). These are concatenated and passed through further MLP layers (`fc2`, `fc3`) to produce the final score matrix (`log_alpha`).
    *   The rest of the training (Gumbel-Sinkhorn) and evaluation (Hungarian algorithm) process remains similar to the first implementation.

## Core Concepts

*   **Gumbel-Sinkhorn:** A differentiable technique to approximate sampling from the space of permutation matrices. It involves adding Gumbel noise to a score matrix (`log_alpha`), scaling by a temperature, and applying Sinkhorn normalization (iterative row/column normalization). This produces a "soft" doubly stochastic matrix used for calculating a differentiable loss during training.
*   **Permutation Loss:** The training objective is typically an L2 (MSE) loss comparing the ground truth sorted sequence with the sequence obtained by applying the inferred (soft or hard) permutation to the original unsorted input.
*   **Matching/Hungarian Algorithm:** Used during evaluation (when differentiability is not needed) to find the optimal hard permutation matrix corresponding to the network's output scores (`log_alpha`) by solving the linear assignment problem.
*   **Deep Sets (Context Model):** A permutation-invariant architecture used in `Sorting_With_Context/` to generate a context vector that summarizes the entire input set/sequence, allowing the network to potentially make more informed sorting decisions for each element.

## Usage

Navigate into the desired implementation directory (`Sorting_Same_As_Paper/` or `Sorting_With_Context/`) and run the script:

```bash
cd Sorting_Same_As_Paper # or Sorting_With_Context
python sort.py