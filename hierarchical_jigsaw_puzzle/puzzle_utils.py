import numpy as np
import torch

def np_divide_image(image: np.ndarray, num_pieces: int):
    height, width = image.shape[-2:]
    piece_height = height // num_pieces
    piece_width  = width // num_pieces
    pieces = []
    for p_h in range(num_pieces):
        for p_w in range(num_pieces):
            left   = p_w * piece_width
            right  = left + piece_width
            top    = p_h * piece_height
            bottom = top + piece_height
            piece  = image[:,top:bottom,left:right]
            pieces.append(piece)
    permute_index = np.random.permutation(num_pieces**2)
    pieces = np.stack(pieces, 0) # (num_pieces, channels, height//num_pieces, width//num_pieces)
    random_pieces = pieces[permute_index]
    return (pieces, random_pieces, permute_index)

def tch_divide_image(image: torch.Tensor, num_pieces: int):
    height, width = image.shape[-2:]
    piece_height = height // num_pieces
    piece_width  = width // num_pieces
    pieces = []
    for p_h in range(num_pieces):
        for p_w in range(num_pieces):
            left   = p_w * piece_width
            right  = left + piece_width
            top    = p_h * piece_height
            bottom = top + piece_height
            piece  = image[:,top:bottom,left:right]
            pieces.append(piece)
    permute_index = torch.randperm(num_pieces**2)
    pieces = torch.stack(pieces, 0) # (num_pieces, channels, height//num_pieces, width//num_pieces)
    random_pieces = pieces[permute_index]
    return (pieces, random_pieces, permute_index)

def batch_tch_divide_image(images: torch.Tensor, num_pieces: int):
    batch_pieces, batch_random_pieces, batch_permute_index = [], [], []
    for image in images:
        pieces, random_pieces, permute_index = tch_divide_image(image, num_pieces)
        batch_pieces.append(pieces); batch_random_pieces.append(random_pieces); batch_permute_index.append(permute_index)
    return torch.stack(batch_pieces, 0), torch.stack(batch_random_pieces, 0), torch.stack(batch_permute_index, 0)

def reconstruct_image_from_groups(patch_tensor):
    """
    Converts a batch of patches to full images.
    
    Args:
        patch_tensor: Tensor of shape [n, A, 1, B, B] where
                      A is a perfect square (number of patches per image)
    
    Returns:
        images: Tensor of shape [n, 1, sqrt(A)*B, sqrt(A)*B]
    """
    n, A, c, B, _ = patch_tensor.shape
    S = int(A ** 0.5)
    assert S * S == A, "A (number of patches) must be a perfect square"

    return (patch_tensor
            .reshape(n, S, S, c, B, B)      # [n, S_row, S_col, c, B, B]
            .permute(0, 3, 1, 4, 2, 5)     # [n, c, S_row, B, S_col, B]
            .reshape(n, c, S * B, S * B))


def get_groups(patches, batch_size, num_groups=7):
    """
    Group patches for a whole batch based on mean intensity.

    Args:
        patches: Tensor of shape (B * N, C, H, W)
        batch_size: Number of samples in the batch (B)
        num_groups: Number of groups to form

    Returns:
        A LongTensor of shape (B, num_groups, N // num_groups) containing indices of patches in each group
    """
    N = patches.shape[0] // batch_size  # total patches per image
    patch_means = patches.view(batch_size, N, -1).mean(dim=2)  # (B, N)
    sorted_indices = patch_means.argsort(dim=1)  # (B, N)

    groups = sorted_indices.view(batch_size, num_groups, N // num_groups)  # (B, G, N//G)
    return groups


def model_based_grouping(patches, batch_size, num_groups, model):
    """
    Group patch indices using a trainable model.

    Args:
        patches: Tensor of shape (B*N, 1, 4, 4)
        batch_size: B
        num_groups: G
        model: PatchGroupingNet instance

    Returns:
        Tensor of shape (B, G, N//G) with grouped indices
    """
    with torch.no_grad():
        logits = model(patches)              # (B*N, G)
        pred_groups = logits.argmax(dim=1)   # (B*N,)

    N = patches.shape[0] // batch_size
    group_indices = [[] for _ in range(batch_size * num_groups)]

    for i in range(patches.shape[0]):
        b = i // N
        g = pred_groups[i].item()
        group_indices[b * num_groups + g].append(i)

    # Convert to tensor with padding if needed
    group_tensor = torch.zeros((batch_size, num_groups, N // num_groups), dtype=torch.long)
    for b in range(batch_size):
        for g in range(num_groups):
            idx = group_indices[b * num_groups + g]
            if len(idx) < N // num_groups:
                idx += [idx[0]] * (N // num_groups - len(idx))  # pad with copies
            group_tensor[b, g] = torch.tensor(idx[:N // num_groups])

    return group_tensor




