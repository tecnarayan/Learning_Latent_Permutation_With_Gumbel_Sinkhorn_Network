import torch
import time
import matplotlib.pyplot as plt
import numpy as np

# Set seed for reproducibility
torch.manual_seed(0)

# Dimensions
ns, nt = 1000, 1000

# Dense vector
x = torch.rand(nt)

# Range of masking probabilities (sparsity levels)
probs = [1e-3, 5e-3, 1e-2] #[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

dense_times = []
sparse_times = []

for p in probs:
    # Generate kernel and mask
    K = torch.rand(ns, nt)
    mask = torch.bernoulli(torch.full((ns, nt), p)).bool()

    # Sparsify K
    K_spar = torch.zeros_like(K)
    K_spar[mask] = K[mask]

    # Create sparse COO tensor
    indices = mask.nonzero(as_tuple=False).T
    values = K_spar[mask]
    K_sparse = torch.sparse_coo_tensor(indices, values, size=K.shape).coalesce()

    # Dense × dense
    start_dense = time.time()
    dense_result = torch.matmul(K_spar, x)
    end_dense = time.time()
    dense_times.append(end_dense - start_dense)

    # Sparse × dense
    start_sparse = time.time()
    sparse_result = torch.sparse.mm(K_sparse, x.unsqueeze(1)).squeeze()
    end_sparse = time.time()
    sparse_times.append(end_sparse - start_sparse)

    # Optional: check correctness
    error = torch.norm(dense_result - sparse_result).item()
    print(f"p={p:.0e} | error={error:.2e} | dense={dense_times[-1]:.4f}s | sparse={sparse_times[-1]:.4f}s")

# -------------------------------
# Plotting
# -------------------------------
plt.figure(figsize=(8, 5))
plt.plot(probs, dense_times, marker='o', label='Dense × Dense')
plt.plot(probs, sparse_times, marker='s', label='Sparse × Dense')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Masking Probability (log scale)')
plt.ylabel('Time (s, log scale)')
plt.title('Matrix-vector multiplication times vs. sparsity')
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.show()
