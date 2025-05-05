import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
import time
import math

# --- Hyperparameters ---
N_NUMBERS = 200
BATCH_SIZE = 64*4
LEARNING_RATE = 0.001
TEMPERATURE = 1.0       # Gumbel-Sinkhorn 
# for N = 200 .. N_ITER = 4 .. N_UNIT = 64  .. noise is 0...  (T,A) (10,0.03) ( 5 , 0.1) (1 , 0) (0.5 , ) (0.1,still dead ) (0.01, dead )(0.001 , dead )
NOISE_FACTOR = 1.0      # Gumbel noise scaling during training (0.0 for deterministic Sinkhorn)
N_ITER_SINKHORN = 4 #20//5    # Sinkhorn iterations
N_UNITS = 64            # Hidden units in the network (used for embedding and scoring)
NUM_ITERATIONS = 8000 * 3   # Training iterations
EVAL_BATCH_SIZE = 100*10   # Batch size for final evaluation
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
# [Keep all helper functions exactly the same as before]
def sample_gumbel(shape, eps=1e-20, device='cpu'):
    """Samples arbitrary-shaped standard gumbel variables."""
    u = torch.rand(shape, dtype=torch.float32, device=device)
    return -torch.log(-torch.log(u + eps) + eps)

def sinkhorn(log_alpha, n_iters=20):
    """Performs incomplete Sinkhorn normalization."""
    if log_alpha.dim() == 2:
        log_alpha = log_alpha.unsqueeze(0)
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
    return torch.exp(log_alpha)

def gumbel_sinkhorn(log_alpha, temp=1.0, noise_factor=1.0, n_iters=20, device='cpu'):
    """Applies Gumbel noise and Sinkhorn normalization."""
    if log_alpha.dim() == 2:
        log_alpha = log_alpha.unsqueeze(0)
    batch_size, n, _ = log_alpha.shape

    if noise_factor > 0:
        gumbel = sample_gumbel(log_alpha.shape, device=device) * noise_factor
        log_alpha_w_noise = log_alpha + gumbel
    else:
        log_alpha_w_noise = log_alpha

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
    return ordered, random, permutations.long()


# --- MODIFIED Model Definition with Deep Sets Context ---
class SortingNetworkWithDeepSetContext(nn.Module):
    def __init__(self, n_numbers, n_units):
        super().__init__()
        self.n_numbers = n_numbers
        self.n_units = n_units # Keep n_units consistent

        # MLP 'g' for embedding each element x_i -> h_i
        # (Using ec1, ec2 as per your comment)
        self.ec1 = nn.Linear(1, n_units)
        self.ec2 = nn.Linear(n_units, n_units)

        # MLP 'f' applied to the aggregated sum 's' -> context 'c'
        # (Using ec1_, ec2_ as per your comment)
        self.ec1_ = nn.Linear(n_units, n_units) # Input is sum of n_units embeddings
        self.ec2_ = nn.Linear(n_units, n_units) # Output is context of n_units

        # Original MLP layers for processing individual elements,
        # now potentially combining with context
        self.fc1 = nn.Linear(1, n_units) # Processes original x_i

        # This layer now needs to combine the output of fc1 (n_units)
        # and the context from ec2_ (n_units). Input size is n_units * 2.
        self.fc2 = nn.Linear(n_units * 2, n_units)

        # Final output layer remains the same
        self.fc3 = nn.Linear(n_units, n_numbers)

    def forward(self, random_input):
        """Input shape: [batch_size, n_numbers]"""
        batch_size = random_input.shape[0]

        # Reshape for element-wise processing: [batch_size * n_numbers, 1]
        x_flat = random_input.reshape(-1, 1)

        # --- Deep Sets Context Calculation ---
        # 1. Apply MLP 'g' (ec1, ec2) to each element x_i -> h_i
        h_i_flat = F.relu(self.ec2(F.relu(self.ec1(x_flat))))
        # Reshape back to sequence format: [batch_size, n_numbers, n_units]
        h_elements = h_i_flat.reshape(batch_size, self.n_numbers, self.n_units)

        # 2. Calculate sum 's' over embeddings h_i
        # Summation over the sequence dimension (dim=1)
        s = torch.sum(h_elements, dim=1) # Shape: [batch_size, n_units]

        # 3. Apply MLP 'f' (ec1_, ec2_) to the sum 's' -> context 'c'
        context_c = F.relu(self.ec2_(F.relu(self.ec1_(s)))) # Shape: [batch_size, n_units]
        # --- End Context Calculation ---

        # --- Combine Element Representation with Context ---
        # 1. Process original x_i with fc1
        # x_flat is already [batch_size * n_numbers, 1]
        x_processed_fc1_flat = F.relu(self.fc1(x_flat)) # Shape: [batch_size * n_numbers, n_units]

        # 2. Expand context 'c' to match the flattened element dimension
        # context_c shape: [batch_size, n_units]
        # Need shape: [batch_size, n_numbers, n_units] then flatten
        context_c_expanded = context_c.unsqueeze(1).expand(-1, self.n_numbers, -1)
        # Flatten context: [batch_size * n_numbers, n_units]
        context_c_flat = context_c_expanded.reshape(-1, self.n_units)

        # 3. Concatenate the processed element and the context
        # Shape: [batch_size * n_numbers, n_units * 2]
        combined_input_for_fc2 = torch.cat((x_processed_fc1_flat, context_c_flat), dim=1)

        # --- Final Scoring Layers ---
        # 4. Pass combined representation through fc2
        x_processed_fc2 = F.relu(self.fc2(combined_input_for_fc2)) # Shape: [batch_size * n_numbers, n_units]

        # 5. Pass through fc3 to get final scores
        # Shape: [batch_size * n_numbers, n_numbers]
        processed = self.fc3(x_processed_fc2)

        # 6. Reshape to log_alpha matrix
        # Shape: [batch_size, n_numbers, n_numbers]
        log_alpha = processed.reshape(batch_size, self.n_numbers, self.n_numbers)
        return log_alpha

# --- Training Function ---
# [Keep the training function exactly the same as before]
def train_model(model, optimizer, num_iterations, hparams, device):
    print("Starting training...")
    model.train()
    start_time = time.time()

    loss_window = [100000.0] * 8

    for i in range(num_iterations):
        # 1. Generate Data
        ordered_gt, random_input, _ = sample_uniform_and_order(
            hparams['batch_size'], hparams['n_numbers'], hparams['prob_inc'], device=device
        )

        # 2. Forward Pass -> Get log_alpha
        log_alpha = model(random_input) # Model forward pass now includes context

        # 3. Gumbel-Sinkhorn -> Get Soft Permutation Matrix
        soft_perms_inf, _ = gumbel_sinkhorn(
            log_alpha,
            temp=hparams['temperature'],
            noise_factor=hparams['noise_factor'],
            n_iters=hparams['n_iter_sinkhorn'],
            device=device
        )

        # 4. Calculate L2 Loss using Soft Permutation
        inv_soft_perms = soft_perms_inf.transpose(-1, -2)
        random_input_rs = random_input.unsqueeze(2)
        ordered_gt_rs = ordered_gt.unsqueeze(2)
        ordered_inf_soft = torch.matmul(inv_soft_perms, random_input_rs)
        loss = F.mse_loss(ordered_inf_soft, ordered_gt_rs)


        

        # 5. Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 6. Logging
        if (i + 1) % 500 == 0:
            elapsed_time = time.time() - start_time
            print(f"Iteration [{i+1}/{num_iterations}], Loss: {loss.item():.6f}, Time: {elapsed_time:.2f}s")
            start_time = time.time() # Reset timer

            ######################
            current_loss = loss.item()
            loss_window.pop(0)           # Remove first (oldest) element
            loss_window.append(current_loss)  # Add new loss at the end
            if sum(loss_window[:4]) <= sum(loss_window[4:]):
                print("Loss decreasing sufficiently; breaking out of the loop.")
                break
            ######################

        # if ( i+1 & 2000) == 0:
        #     evaluate_model(model, hparams, DEVICE)

    print("Training finished.")


# --- Evaluation Function ---
# [Keep the evaluation function exactly the same as before]
def evaluate_model(model, hparams, device):
    print("\nStarting evaluation...")
    model.eval()
    total_correct_elements = 0
    total_elements = 0
    total_perfect_sequences = 0
    total_sequences = 0
    all_mae = []

    with torch.no_grad():
        # Calculate number of batches needed based on eval_batch_size
        num_eval_batches = math.ceil(hparams['eval_batch_size'] / hparams['batch_size'])

        for _ in range(num_eval_batches): # Evaluate enough batches
            # 1. Generate Test Data
            # Use actual batch_size for generation, total samples will be approx eval_batch_size
            current_batch_size = min(hparams['batch_size'], hparams['eval_batch_size'] - (_ * hparams['batch_size']))
            if current_batch_size <= 0: continue # Avoid empty batch

            ordered_gt, random_input, _ = sample_uniform_and_order(
                current_batch_size, hparams['n_numbers'], hparams['prob_inc'], device=device
            )

            # 2. Forward Pass -> Get log_alpha
            log_alpha = model(random_input) # Model forward pass now includes context

            # 3. Get Hard Permutation using Matching
            hard_perms_inf = matching(log_alpha)
            hard_perms_inf = hard_perms_inf.to(device)

            # 4. Apply Inferred Hard Permutation
            inverse_hard_perms_inf = invert_listperm(hard_perms_inf)
            random_input_rs = random_input.unsqueeze(2)
            ordered_inf_hard = permute_batch_split(random_input_rs, inverse_hard_perms_inf).squeeze(2)

            # 5. Calculate Metrics
            correct_elements = torch.isclose(ordered_inf_hard, ordered_gt, atol=1e-4).sum().item()
            perfect_sequences = torch.all(torch.isclose(ordered_inf_hard, ordered_gt, atol=1e-4), dim=1).sum().item()
            mae = torch.abs(ordered_inf_hard - ordered_gt).mean().item()

            total_correct_elements += correct_elements
            total_elements += ordered_gt.numel()
            total_perfect_sequences += perfect_sequences
            total_sequences += ordered_gt.shape[0]
            all_mae.append(mae)

            # Optional: Print one example comparison per eval batch
            # print("\n--- Example Prediction (Eval Batch) ---")
            # idx = 0 # Show first example of the current batch
            # print(f"Random Input: {random_input[idx].cpu().numpy().round(3)}")
            # print(f"True Sorted:  {ordered_gt[idx].cpu().numpy().round(3)}")
            # print(f"Pred Sorted:  {ordered_inf_hard[idx].cpu().numpy().round(3)}")
            # print("-" * 35)


    avg_mae = np.mean(all_mae) if all_mae else 0
    element_accuracy = total_correct_elements / total_elements if total_elements > 0 else 0
    sequence_accuracy = total_perfect_sequences / total_sequences if total_sequences > 0 else 0

    print("\n--- Evaluation Results ---")
    print(f"Total sequences evaluated: {total_sequences}")
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
        'eval_batch_size': EVAL_BATCH_SIZE, # Total samples for evaluation
        'prob_inc': PROB_INC,
    }

    # Instantiate MODIFIED Model and Optimizer
    model = SortingNetworkWithDeepSetContext(hparams['n_numbers'], hparams['n_units']).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=hparams['learning_rate'])

    # Train the Model
    train_model(model, optimizer, hparams['num_iterations'], hparams, DEVICE)

    # Evaluate the Model
    evaluate_model(model, hparams, DEVICE)