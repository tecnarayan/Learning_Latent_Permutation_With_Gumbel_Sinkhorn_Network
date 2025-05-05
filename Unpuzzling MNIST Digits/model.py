import torch
import torch.nn as nn

class ConvModel(nn.Module):
    def __init__(
        self,
        in_c: int,
        pieces: int,
        image_size: int,
        hid_c: int = 64,
        stride: int = 2,
        kernel_size: int = 5
    ):
        super().__init__()
        self.g_1 = nn.Sequential(
            nn.Conv2d(in_c, hid_c, kernel_size, padding=kernel_size//2),
            nn.ReLU(True),
            nn.MaxPool2d(stride),
        )
        # self.g_2 = nn.Linear((image_size**2//(pieces**2))//(stride**2)*64, pieces**2, bias=False)
        patch_size = image_size // pieces
        feature_map_size = patch_size // stride  # after MaxPool
        flattened_size = hid_c * feature_map_size * feature_map_size

        self.g_2 = nn.Linear(flattened_size, pieces**2, bias=False)

        print()
        nn.init.kaiming_normal_(self.g_1[0].weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.g_1[0].bias, 0)
        nn.init.kaiming_normal_(self.g_2.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, pieces):
        # print("shape of pieces : ", pieces.shape)
        batch_size = pieces.shape[0]
        pieces = pieces.transpose(0,1).contiguous() # (num_pieces, batchsize, channels, height, width)
        pieces = [self.g_1(p).reshape(batch_size, -1) for p in pieces] # convolve and vectorize
        latent = [self.g_2(p) for p in pieces]
        latent_matrix = torch.stack(latent, 1)
        # print("shape of the latent matrix : ", latent_matrix.shape)
        return latent_matrix

class Vectorize(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        return x.reshape(x.shape[0], self.channels)

# import torch
# import torch.nn as nn

# class ConvModel(nn.Module):
#     def __init__(
#         self,
#         in_c: int,
#         pieces: int,
#         image_size: int,
#         hid_c: int = 64,
#         stride: int = 2,
#         kernel_size: int = 5
#     ):
#         super().__init__()
#         self.g_1 = nn.Sequential(
#             nn.Conv2d(in_c, hid_c, kernel_size, padding=kernel_size // 2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(stride)
#         )

#         self.g_2 = None  # will be initialized dynamically
#         self.hid_c = hid_c
#         self.stride = stride
#         self.image_size = image_size
#         self.pieces = pieces

#         # Initialize g_1 layers
#         nn.init.kaiming_normal_(self.g_1[0].weight, mode="fan_out", nonlinearity="relu")
#         nn.init.constant_(self.g_1[0].bias, 0)

#     def forward(self, pieces):
#         # pieces: (batch_size, num_pieces, channels, height, width)
#         # print("shape of pieces : ", pieces.shape)

#         batch_size, num_pieces = pieces.shape[:2]
#         device = pieces.device
#         conv_outs = []

#         for i in range(batch_size):
#             p = pieces[i]  # shape: (num_pieces, channels, height, width)

#             out = self.g_1(p)  # -> (num_pieces, hid_c, H', W')
#             out = out.view(num_pieces, -1)  # (num_pieces, hid_c * H' * W')

#             if self.g_2 is None:
#                 print("Hello world!########################################################################################")
#                 input_dim = out.shape[1]
#                 linear_layer = nn.Linear(input_dim, num_pieces, bias=False)
#                 nn.init.kaiming_normal_(linear_layer.weight, mode="fan_out", nonlinearity="relu")
#                 self.g_2 = linear_layer.to(device)

#             conv_outs.append(self.g_2(out))  # (num_pieces, num_pieces)

#         latent_matrix = torch.stack(conv_outs, dim=0)  # (batch_size, num_pieces, num_pieces)
#         # print("shape of the latent matrix : ", latent_matrix.shape)
#         return latent_matrix


# class Vectorize(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.channels = channels

#     def forward(self, x):
#         return x.reshape(x.shape[0], self.channels)
