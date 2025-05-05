import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
import time
import math

# --- Hyperparameters ---
N_NUMBERS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
TEMPERATURE = 1.0       # Gumbel-Sinkhorn temperature
NOISE_FACTOR = 1.0      # Gumbel noise scaling during training (0.0 for deterministic Sinkhorn)
N_ITER_SINKHORN = 20 # 10 # 20    # Sinkhorn iterations
N_UNITS = 32 # 64            # Hidden units in the network
NUM_ITERATIONS = 8000   # Training iterations
EVAL_BATCH_SIZE = 100   # Batch size for final evaluation
PROB_INC = 1.0          # Probability of generating increasing sequences (1.0 means always increasing)
SEED = 42

# --- Device Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Seed for Reproducibility ---
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Helper Functions (Sinkhorn Ops, Data Gen, Matching) ---

def sample_gumbel(shape, eps=1e-20, device='cpu'):
    """Samples arbitrary-shaped standard gumbel variables."""
    u = torch.rand(shape, dtype=torch.float32, device=device)
    return -torch.log(-torch.log(u + eps) + eps) #  noise is   -log(-log[U]) .. where U ~ (0,1)

def sinkhorn(log_alpha, n_iters=20):
    """Performs incomplete Sinkhorn normalization."""
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True) # log(e^a / e^a + e^b) == [ a - log(e^a + e^b)]
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
    return torch.exp(log_alpha)

def gumbel_sinkhorn(log_alpha, temp=1.0, noise_factor=1.0, n_iters=20, device='cpu'):
    """Applies Gumbel noise and Sinkhorn normalization."""
    # noise is added as it helps in convergence .. but why ????
    # log_alpha is bacially log of score matrix .. shape -> [batch_size , N , N] 

    gumbel = sample_gumbel(log_alpha.shape, device=device) * noise_factor
    log_alpha_w_noise = log_alpha + gumbel

    log_alpha_w_noise = log_alpha_w_noise / temp
    sink_matrix = sinkhorn(log_alpha_w_noise, n_iters)
    return sink_matrix, log_alpha_w_noise # Return both soft matrix and noisy logits

def matching(matrix_batch):
    """Solves batch matching problem using SciPy (expects CPU tensor)."""
    matrix_batch_np = matrix_batch.detach().cpu().numpy()
    if matrix_batch_np.ndim == 2:
        matrix_batch_np = np.reshape(matrix_batch_np, [1, matrix_batch_np.shape[0], matrix_batch_np.shape[1]])
    sol = np.zeros((matrix_batch_np.shape[0], matrix_batch_np.shape[1]), dtype=np.int64)
    for i in range(matrix_batch_np.shape[0]):
        row_ind, col_ind = linear_sum_assignment(-matrix_batch_np[i, :, :])
        sol[i, row_ind] = col_ind.astype(np.int64)
    return torch.from_numpy(sol).long()

def invert_listperm(listperm):
    """Inverts a batch of permutations efficiently."""
    batch_size, n_objects = listperm.shape
    device = listperm.device
    listperm = listperm.long()
    indices = torch.arange(n_objects, device=device).unsqueeze(0).expand(batch_size, -1)
    inv_listperm = torch.empty_like(listperm)
    inv_listperm.scatter_(dim=1, index=listperm, src=indices)
    return inv_listperm.long()

def permute_batch_split(batch_split, permutations):
    """Permutes batch using gather."""
    batch_size, n_objects, object_size = batch_split.shape
    device = batch_split.device
    permutations = permutations.long().to(device)
    perm_expanded = permutations.unsqueeze(2).expand(-1, -1, object_size)
    perm_batch_split = torch.gather(batch_split, 1, perm_expanded)
    return perm_batch_split

def sample_uniform_and_order(n_lists, n_numbers, prob_inc, device='cpu'):
    """Samples uniform random numbers [0,1] and sorts them."""
    bern = torch.bernoulli(torch.full((n_lists, 1), prob_inc, device=device))
    sign = (bern * 2 - 1).float()
    random = torch.rand(n_lists, n_numbers, dtype=torch.float32, device=device)
    random_with_sign = random * sign
    # Use sort instead of topk for potentially better numerical stability with floats
    ordered_signed, permutations = torch.sort(random_with_sign, dim=1, descending=False) # Sort ascending based on sign trick
    ordered = ordered_signed * sign # Undo sign trick

    # Need to get the permutation P such that random[b, P[b, i]] = ordered[b, i]
    # `permutations` from torch.sort gives indices into the *original* tensor
    # such that random_with_sign[b, permutations[b, i]] = ordered_signed[b, i]
    # This `permutations` is what we need (maps sorted index to original index).
    return ordered, random, permutations.long()

# --- Model Definition ---
class SortingNetwork(nn.Module):
    def __init__(self, n_numbers, n_units):
        super().__init__()
        self.n_numbers = n_numbers # N
        self.fc1 = nn.Linear(1, n_units)
        self.fc2 = nn.Linear(n_units, n_units) # Added another layer
        self.fc3 = nn.Linear(n_units, n_numbers) # Output layer

    def forward(self, random_input):
        """Input shape: [batch_size, n_numbers]"""
        batch_size = random_input.shape[0]
        x = random_input.reshape(-1, 1)  # Reshape for element-wise processing: [batch_size * n_numbers, 1]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        processed = self.fc3(x) # Output 'scores' for each position: [batch_size * n_numbers, n_numbers]
        # Reshape to log_alpha matrix: [batch_size, n_numbers, n_numbers]
        # Interpretation: log_alpha[b, i, j] is score for element i ending up in position j
        log_alpha = processed.reshape(batch_size, self.n_numbers, self.n_numbers)
        return log_alpha


def train_model(model, optimizer, num_iterations, hparams, device):
    print("Starting training...")
    model.train()
    start_time = time.time()
    for i in range(num_iterations):
        
        ordered_gt, random_input, _ = sample_uniform_and_order(
            hparams['batch_size'], hparams['n_numbers'], hparams['prob_inc'], device=device
        )

        # print(ordered_gt.shape) -> [batch_size , N]
        # .. similarly ... random_input.shape is [batch_szie  , N]

        log_alpha = model(random_input) # shae is [batch_size , N , N] .. its a matrix which we got .. log(score matrix) 

        soft_perms_inf, _ = gumbel_sinkhorn(
            log_alpha,
            temp=hparams['temperature'],
            noise_factor=hparams['noise_factor'],
            n_iters=hparams['n_iter_sinkhorn'],
            device=device
        )

        # 4. Calculate L2 Loss using Soft Permutation
        # We want to check if applying the *inverse* soft perm to random_input gives ordered_gt
        # Inverse of soft perm matrix is its transpose
        inv_soft_perms = soft_perms_inf.transpose(-1, -2) # Shape: [batch_size, N, N] # tranpote for last two dimention

        # Reshape inputs for matmul: [batch_size, N, 1]
        random_input_rs = random_input.unsqueeze(2) # add 1 to 2th position in shape 
        ordered_gt_rs = ordered_gt.unsqueeze(2)

        # Apply inverse permutation: P^T * random = inferred_ordered
        ordered_inf_soft = inv_soft_perms @ random_input_rs # Shape: [batch_size, N, 1]

        loss = F.mse_loss(ordered_inf_soft, ordered_gt_rs) # Mean Squared Error

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 500 == 0:
            elapsed_time = time.time() - start_time
            print(f"Iteration [{i+1}/{num_iterations}], Loss: {loss.item():.6f}, Time: {elapsed_time:.2f}s")
            start_time = time.time() # Reset timer

    print("Training finished.")

# --- Evaluation Function ---
def evaluate_model(model, hparams, device):
    print("\nStarting evaluation...")
    model.eval()
    total_correct_elements = 0
    total_elements = 0
    total_perfect_sequences = 0
    total_sequences = 0
    all_mae = []

    with torch.no_grad():
        for _ in range(hparams['eval_batch_size'] // hparams['batch_size'] + 1): # Evaluate enough batches
            # 1. Generate Test Data
            ordered_gt, random_input, _ = sample_uniform_and_order(
                hparams['batch_size'], hparams['n_numbers'], hparams['prob_inc'], device=device
            )

            # 2. Forward Pass -> Get log_alpha
            log_alpha = model(random_input)

            # 3. Get Hard Permutation using Matching (NO Gumbel noise for eval)
            # We apply Sinkhorn deterministically (temp=1, noise=0) or just use log_alpha
            # Using log_alpha directly with matching is equivalent to finding the optimal permutation
            # based on the network scores, which is standard for evaluation.
            # _, log_alpha_eval = gumbel_sinkhorn(log_alpha, temp=1.0, noise_factor=0.0, n_iters=hparams['n_iter_sinkhorn'], device=device)

            # Perform matching on the raw scores (log_alpha) on CPU
            hard_perms_inf = matching(log_alpha) # Returns CPU tensor [batch_size, N]
            hard_perms_inf = hard_perms_inf.to(device) # Move back to device

            # 4. Apply Inferred Hard Permutation
            # We need the permutation that maps original indices to sorted positions.
            # `hard_perms_inf[b, i]` gives the *sorted position* for original element `i`.
            # To reconstruct the sorted list, we need the *inverse* permutation:
            # `inv_perm[b, j]` gives the *original index* of the element that belongs in sorted position `j`.
            inverse_hard_perms_inf = invert_listperm(hard_perms_inf)

            # Apply inverse perm to random_input to get predicted sorted order
            # Input needs shape [batch_size, N, 1] for permute_batch_split
            random_input_rs = random_input.unsqueeze(2)
            ordered_inf_hard = permute_batch_split(random_input_rs, inverse_hard_perms_inf).squeeze(2) # Shape: [batch_size, N]

            # 5. Calculate Metrics
            # Element-wise comparison (allow small tolerance for float comparisons)
            correct_elements = torch.isclose(ordered_inf_hard, ordered_gt, atol=1e-4).sum().item()
            perfect_sequences = torch.all(torch.isclose(ordered_inf_hard, ordered_gt, atol=1e-4), dim=1).sum().item()
            mae = torch.abs(ordered_inf_hard - ordered_gt).mean().item()

            total_correct_elements += correct_elements
            total_elements += ordered_gt.numel()
            total_perfect_sequences += perfect_sequences
            total_sequences += ordered_gt.shape[0]
            all_mae.append(mae)

            # Optional: Print one example comparison
            #if _ == 0:
            print("\n--- Example Prediction ---")
            idx = 0
            print(f"Random Input: {random_input[idx].cpu().numpy().round(3)}")
            print(f"True Sorted:  {ordered_gt[idx].cpu().numpy().round(3)}")
            print(f"Pred Sorted:  {ordered_inf_hard[idx].cpu().numpy().round(3)}")
            print("-" * 26)


    avg_mae = np.mean(all_mae)
    element_accuracy = total_correct_elements / total_elements if total_elements > 0 else 0
    sequence_accuracy = total_perfect_sequences / total_sequences if total_sequences > 0 else 0

    print("\n--- Evaluation Results ---")
    print(f"Average Mean Absolute Error (MAE): {avg_mae:.6f}")
    print(f"Element-wise Accuracy: {element_accuracy:.4f} ({total_correct_elements}/{total_elements})")
    print(f"Perfect Sequence Accuracy: {sequence_accuracy:.4f} ({total_perfect_sequences}/{total_sequences})")
    print("-" * 26)

# --- Main Execution ---
if __name__ == "__main__":
    # Hyperparameters Dictionary
    hparams = {
        'n_numbers': N_NUMBERS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'temperature': TEMPERATURE,
        'noise_factor': NOISE_FACTOR,
        'n_iter_sinkhorn': N_ITER_SINKHORN,
        'n_units': N_UNITS,
        'num_iterations': NUM_ITERATIONS,
        'eval_batch_size': EVAL_BATCH_SIZE,
        'prob_inc': PROB_INC,
    }

    # Instantiate Model and Optimizer
    model = SortingNetwork(hparams['n_numbers'], hparams['n_units']).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=hparams['learning_rate'])

    # Train the Model
    train_model(model, optimizer, hparams['num_iterations'], hparams, DEVICE)

    # Evaluate the Model
    evaluate_model(model, hparams, DEVICE)