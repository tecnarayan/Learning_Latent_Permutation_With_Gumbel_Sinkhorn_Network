import torch
import time

# Do speed comparison with classical sinkhorn algorithm.
# Is the computation graph is small ? Does it also speed up the training ?
# Accuracy vs sample density
# Quality of resutls of exp1 vs sample density

def sparsify_kernel(K, a, b, s):
    """
    Poisson-style sparsification of kernel matrix K using marginals a and b.

    Args:
        K:     [ns, nt] Kernel matrix (e.g., exp((log_alpha + gumbel)/tau))
        a:     [ns] source distribution (1D tensor)
        b:     [nt] target distribution (1D tensor)
        s:     expected number of samples (int or float)
        method: 'spar-sink' or 'rand-sink'

    Returns:
        K_spar: sparsified and reweighted kernel [ns, nt]
        mask: binary mask [ns, nt] indicating retained entries
    """
    ns, nt = K.shape
    K = K.float()
    a = a.float()
    b = b.float()

    # For our purpose of doubly stochastic matrices, spar-sink and rand-sink are essentially same. 
    # prob = torch.sqrt(torch.outer(a, b)) * (K > 0)
    prob = torch.log(K)
    prob /= prob.sum()
    prob *= s
    prob = torch.clamp(prob, max=1.0)

    # Bernoulli sampling
    mask = torch.bernoulli(prob)

    # Reweight
    K_spar = torch.zeros_like(K)
    keep = (mask.bool()) & (prob > 0)
    K_spar[keep] = K[keep] / prob[keep]

    return K_spar, mask


def spar_sinkhorn(log_alpha, a, b, s, tau=1.0, n_iter=100, stable=1e-8):
    ns, nt = log_alpha.shape

    a = a.float()
    b = b.float()
    K = torch.exp(log_alpha)

    K_spar_dense, mask = sparsify_kernel(K, a, b, s)

    # Convert to sparse COO format
    indices = torch.nonzero(K_spar_dense, as_tuple=False).T  # shape: [2, nnz]
    values = K_spar_dense[indices[0], indices[1]]
    K_spar = torch.sparse_coo_tensor(indices, values, size=K.shape)
    # K_spar = K_spar_dense
    # u = torch.ones(ns, device=K.device) / ns
    # v = torch.ones(nt, device=K.device) / nt

    # for _ in range(n_iter):
    #     Kv = torch.sparse.mm(K_spar, v.unsqueeze(1)).squeeze() + stable
    #     u = a / Kv
    #     Ktu = torch.sparse.mm(K_spar.transpose(0, 1), u.unsqueeze(1)).squeeze() + stable
    #     v = b / Ktu

    # # Final transport plan: P = diag(u) @ K_spar @ diag(v)
    # u_diag = u[indices[0]]
    # v_diag = v[indices[1]]
    # values = values * u_diag * v_diag
    # P_sparse = torch.sparse_coo_tensor(indices, values, size=(ns, nt))

    # return P_sparse.to_dense()

    for _ in range(n_iter):
        # Row normalization
        row_sums = K_spar.sum(dim=1, keepdim=True).to_dense() + stable
        K_spar = K_spar * (a[:, None] / row_sums)

        # Column normalization
        col_sums = K_spar.sum(dim=0, keepdim=True).to_dense() + stable
        K_spar = K_spar * (b[None, :] / col_sums)

    return K_spar

def log_sinkhorn_norm(log_alpha: torch.Tensor, n_iter: int =20) -> (torch.Tensor,):
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
    
    return log_alpha.exp()

def gumbel_sinkhorn(
    log_alpha: torch.Tensor,
    tau: float = 1.0,
    n_iter: int = 20,
    noise: bool = True,
    method: str = 'Default',
    a: torch.Tensor = None,
    b: torch.Tensor = None,
    s: int = None
) -> torch.Tensor:
    if noise:
        uniform_noise = torch.rand_like(log_alpha)
        gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-20) + 1e-20)
        log_alpha = (log_alpha + gumbel_noise) / tau

    if method == 'Default':
        return log_sinkhorn_norm(log_alpha, n_iter)
    elif method == 'Sparse':
        assert a is not None and b is not None and s is not None, "a, b, s are required for Sparse Sinkhorn"
        return spar_sinkhorn(log_alpha, a, b, s, tau, n_iter)


# --------------------------
# Test block for comparison
# --------------------------
# if __name__ == "__main__":
#     torch.manual_seed(0)

#     ns, nt = 12544, 12544
#     log_alpha = torch.rand(ns, nt)
#     a = torch.ones(ns)
#     b = torch.ones(nt)
#     s = int(0.001 * ns * nt)  # 0.1% sparsity
#     print(s)
#     tau = 1.0
#     n_iter = 10

# #     # Dense method
#     start_dense = time.time()
#     P_dense = gumbel_sinkhorn(log_alpha, tau, n_iter, noise=False, method='Default')
#     end_dense = time.time()

# #     # Sparse method
#     start_sparse = time.time()
#     P_sparse = gumbel_sinkhorn(log_alpha, tau, n_iter, noise=False, method='Sparse', a=a, b=b, s=s)
#     end_sparse = time.time()

#     # print(f"Sparse Gumbel-Sinkhorn Time: {end_sparse - start_sparse:.4f} sec")

# #     # Compute inner products
#     inner_dense = torch.sum(P_dense * log_alpha).item()
#     inner_sparse = torch.sum(P_sparse * log_alpha).item()
#     fractional_diff = abs(inner_dense - inner_sparse) / abs(inner_dense)

# #     # Compare
#     print(f"\n--- Comparison ---")
#     print(f"Dense Gumbel-Sinkhorn Time:  {end_dense - start_dense:.4f} sec")
#     print(f"Sparse Gumbel-Sinkhorn Time: {end_sparse - start_sparse:.4f} sec")
#     print(f"⟨P_dense, log_alpha⟩:  {inner_dense:.4f}")
#     print(f"⟨P_sparse, log_alpha⟩: {inner_sparse:.4f}")
#     print(f"Fractional difference: {fractional_diff:.4%}")


# import torch
# import time
# import matplotlib.pyplot as plt

# def run_experiment(ns=12544, nt=12544, tau=1.0, n_iter=10, sparsity_levels=None):
#     if sparsity_levels is None:
#         sparsity_levels = [0.001, 0.005, 0.01, 0.02, 0.05]  # from 1% to 100%

#     log_alpha = torch.rand(ns, nt)
#     a = torch.ones(ns)
#     b = torch.ones(nt)

#     inner_dense_list = []
#     inner_sparse_list = []
#     fractional_diff_list = []
#     sparsity_list = []

#     # Dense once
#     P_dense = gumbel_sinkhorn(log_alpha, tau, n_iter, noise=False, method='Default')
#     inner_dense = torch.sum(P_dense * log_alpha).item()

#     for frac in sparsity_levels:
#         print("Hello\n")
#         s = int(frac * ns * nt)

#         P_sparse = gumbel_sinkhorn(log_alpha, tau, n_iter, noise=False, method='Sparse', a=a, b=b, s=s)
#         inner_sparse = torch.sum(P_sparse * log_alpha).item()
#         frac_diff = abs(inner_dense - inner_sparse) / abs(inner_dense)

#         sparsity_list.append(frac * 100)  # convert to %
#         inner_dense_list.append(inner_dense)
#         inner_sparse_list.append(inner_sparse)
#         fractional_diff_list.append(frac_diff * 100)

#     return sparsity_list, inner_dense_list, inner_sparse_list, fractional_diff_list


# if __name__ == "__main__":
#     sparsity_levels, dense_vals, sparse_vals, frac_diffs = run_experiment()

#     # Plot 1: Inner Products
#     plt.figure(figsize=(8, 5))
#     plt.plot(sparsity_levels, dense_vals, label="⟨P_dense, log_alpha⟩", linestyle='--', marker='o')
#     plt.plot(sparsity_levels, sparse_vals, label="⟨P_sparse, log_alpha⟩", linestyle='-', marker='x')
#     plt.xlabel("Sparsity (%)")
#     plt.ylabel("Inner Product")
#     plt.title("Sinkhorn Inner Product vs Sparsity")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     # Plot 2: Fractional Difference
#     plt.figure(figsize=(8, 5))
#     plt.plot(sparsity_levels, frac_diffs, label="Fractional Difference", linestyle='-', marker='s', color='red')
#     plt.xlabel("Sparsity (%)")
#     plt.ylabel("Percentage Difference")
#     plt.title("Percentage Difference in ⟨P, log_alpha⟩ vs Sparsity")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

import torch
import matplotlib.pyplot as plt
import numpy as np

def run_experiment(ns=12544, nt=12544, tau=1.0, n_iter=10, sparsity_levels=None, n_trials=10):
    if sparsity_levels is None:
        sparsity_levels = [0.001, 0.005, 0.01, 0.02, 0.05]  # From 0.1% to 5%

    sparsity_percentages = [frac * 100 for frac in sparsity_levels]

    log_alpha = torch.rand(ns, nt)  # Fixed once
    a = torch.ones(ns)
    b = torch.ones(nt)

    # Compute dense only once
    P_dense = gumbel_sinkhorn(log_alpha, tau, n_iter, noise=False, method='Default')
    inner_dense = torch.sum(P_dense * log_alpha).item()

    all_sparse_vals = []
    all_frac_diffs = []

    for frac in sparsity_levels:
        print(f"Running for sparsity {frac * 100:.2f}%")
        s = int(frac * ns * nt)

        trial_sparse_vals = []
        trial_frac_diffs = []

        for trial in range(n_trials):
            P_sparse = gumbel_sinkhorn(log_alpha, tau, n_iter, noise=False, method='Sparse', a=a, b=b, s=s)
            inner_sparse = torch.sum(P_sparse * log_alpha).item()
            frac_diff = abs(inner_dense - inner_sparse) / abs(inner_dense)

            trial_sparse_vals.append(inner_sparse)
            trial_frac_diffs.append(frac_diff * 100)

        all_sparse_vals.append(trial_sparse_vals)
        all_frac_diffs.append(trial_frac_diffs)

    # Convert to numpy
    all_sparse_vals = np.array(all_sparse_vals)  # shape: (len(sparsity_levels), n_trials)
    all_frac_diffs = np.array(all_frac_diffs)

    mean_sparse_vals = np.mean(all_sparse_vals, axis=1)
    std_sparse_vals = np.std(all_sparse_vals, axis=1)

    mean_frac_diffs = np.mean(all_frac_diffs, axis=1)
    std_frac_diffs = np.std(all_frac_diffs, axis=1)

    return sparsity_percentages, inner_dense, mean_sparse_vals, std_sparse_vals, mean_frac_diffs, std_frac_diffs


if __name__ == "__main__":
    sparsity_levels, dense_val, mean_sparse_vals, std_sparse_vals, mean_frac_diffs, std_frac_diffs = run_experiment()

    # Plot 1: Inner Products
    plt.figure(figsize=(8, 5))
    plt.plot(sparsity_levels, [dense_val] * len(sparsity_levels), label="⟨P_dense, log_alpha⟩", linestyle='--', marker='o')
    plt.errorbar(sparsity_levels, mean_sparse_vals, yerr=std_sparse_vals, label="⟨P_sparse, log_alpha⟩", linestyle='-', marker='x', capsize=4)
    plt.xlabel("Sparsity (%)")
    plt.ylabel("Inner Product")
    plt.title("Sinkhorn Inner Product vs Sparsity (Mean ± Std)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 2: Fractional Difference
    plt.figure(figsize=(8, 5))
    plt.errorbar(sparsity_levels, mean_frac_diffs, yerr=std_frac_diffs, label="Fractional Difference", linestyle='-', marker='s', color='red', capsize=4)
    plt.xlabel("Sparsity (%)")
    plt.ylabel("Percentage Difference")
    plt.title("Percentage Difference in ⟨P, log_alpha⟩ vs Sparsity (Mean ± Std)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
