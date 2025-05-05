import numpy as np
from scipy import sparse
import time
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(0)

# Dimensions
ns, nt = 10000, 10000

# Dense vector
x = np.random.rand(nt)

# Range of masking probabilities (sparsity levels)
probs = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

dense_times = []
sparse_times = []

for p in probs:
    # Create dense matrix
    K = np.random.rand(ns, nt)

    # Create binary mask1e-5, 5e-5, 1e-4, 5e-4, 
    mask = np.random.rand(ns, nt) < p

    # Sparsify dense matrix for dense multiplication
    K_spar = np.zeros_like(K)
    K_spar[mask] = K[mask]

    # Create CSR sparse matrix from masked data
    row, col = np.nonzero(mask)
    data = K[row, col]
    K_sparse = sparse.csr_matrix((data, (row, col)), shape=(ns, nt))

    # Dense × dense
    start_dense = time.time()
    dense_result = K_spar @ x
    end_dense = time.time()
    dense_times.append(end_dense - start_dense)

    # Sparse × dense
    start_sparse = time.time()
    sparse_result = K_sparse @ x
    end_sparse = time.time()
    sparse_times.append(end_sparse - start_sparse)

    # Optional: check correctness
    error = np.linalg.norm(dense_result - sparse_result)
    print(f"p={p:.0e} | error={error:.2e} | dense={dense_times[-1]:.4f}s | sparse={sparse_times[-1]:.4f}s")

# -------------------------------
# Plotting
# -------------------------------
plt.figure(figsize=(8, 5))
plt.plot(probs, dense_times, marker='o', label='Dense × Dense (NumPy)')
plt.plot(probs, sparse_times, marker='s', label='Sparse × Dense (SciPy CSR)')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Masking Probability (log scale)')
plt.ylabel('Time (s, log scale)')
plt.title('Matrix-vector multiplication times vs. sparsity (NumPy vs. SciPy CSR)')
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.show()
