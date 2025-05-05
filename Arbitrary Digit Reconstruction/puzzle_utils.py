import numpy as np
import torch
import matplotlib.pyplot as plt

def np_divide_image(image: np.ndarray, num_pieces: int):
    """
    The final permute_index represent that what is the original location of the patch. 
    For example : Permutation index: [ 9  4  1  2  8 14  3  5 10  7 11 15 12  0  6 13]
    It means the zeroth patch is actually the 9th patch of the original image. So on for other.  
    """

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
            piece  = image[:, top:bottom, left:right]
            pieces.append(piece)
    permute_index = np.random.permutation(num_pieces**2)
    pieces = np.stack(pieces, 0)
    random_pieces = pieces[permute_index]
    return (pieces, random_pieces, permute_index)


def tch_divide_image(image: torch.Tensor, num_pieces: int):
    """
    The final permute_index represent that what is the original location of the patch. 
    For example : Permutation index: [ 9  4  1  2  8 14  3  5 10  7 11 15 12  0  6 13]
    It means the zeroth patch is actually the 9th patch of the original image. So on for other.  
    """

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



# Function to reconstruct image from patches
def reconstruct_image_from_pieces(pieces, perm_index, num_pieces, height, width):
    """
    Reconstructs an image from its patches using the permutation index.
    
    :param pieces: Tensor of shape (num_pieces^2, channels, patch_height, patch_width)
    :param perm_index: Permutation index that indicates the order of patches
    :param num_pieces: Number of pieces along one dimension (e.g., 2 for a 2x2 grid)
    :param height: The height of the original image
    :param width: The width of the original image
    
    :return: Reconstructed image (channels, height, width)
    """
    piece_height = height // num_pieces  # Height of each patch
    piece_width = width // num_pieces  # Width of each patch
    
    # Initialize an empty tensor to hold the reconstructed image
    reconstructed_image = torch.zeros(1, height, width)  # (channels, height, width)
    
    # Loop through all patches and place them in the correct locations
    for i in range(num_pieces**2):
        p_h = i // num_pieces  # Row position of the patch in the grid
        p_w = i % num_pieces   # Column position of the patch in the grid
        
        patch = pieces[i]  # Get the patch based on the permutation index
        
        top = p_h * piece_height
        left = p_w * piece_width
        
        reconstructed_image[:, top:top + piece_height, left:left + piece_width] = patch

    return reconstructed_image
