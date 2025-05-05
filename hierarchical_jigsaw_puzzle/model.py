import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.g_2 = nn.Linear(image_size**2//(stride**2*pieces**2)*64, pieces**2, bias=False)

        nn.init.kaiming_normal_(self.g_1[0].weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.g_1[0].bias, 0)
        nn.init.kaiming_normal_(self.g_2.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, pieces):
        batch_size = pieces.shape[0]
        pieces = pieces.transpose(0,1).contiguous() # (num_pieces, batchsize, channels, height, width)
        pieces = [self.g_1(p).reshape(batch_size, -1) for p in pieces] # convolve and vectorize
        latent = [self.g_2(p) for p in pieces]
        latent_matrix = torch.stack(latent, 1)
        return latent_matrix


class PatchGroupingNet(nn.Module):
    def __init__(self, patch_size=4, num_groups=8):
        super().__init__()
        self.num_groups = num_groups
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (B*N, 16, 4, 4)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),               # (B*N, 16, 1, 1)
            nn.Flatten(),                               # (B*N, 16)
            nn.Linear(16, num_groups)                   # (B*N, G)
        )

    def forward(self, patches):
        return self.encoder(patches)  #

    
class PatchModel(nn.Module):
    def __init__(self, in_c: int, hid_c: int = 64, kernel_size: int = 5, stride: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, hid_c, kernel_size, padding=kernel_size // 2),
            nn.ReLU(True),
            nn.MaxPool2d(stride),
        )
        self.linear = nn.Linear(hid_c * 4 * 4, 49, bias=False)  # assuming 4x4 output for 7x7 patches

        nn.init.kaiming_normal_(self.conv[0].weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.conv[0].bias, 0)
        nn.init.kaiming_normal_(self.linear.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, patches):
        B, N, C, H, W = patches.size()
        patches = patches.view(B * N, C, H, W)
        feats = self.conv(patches)
        feats = feats.view(feats.size(0), -1)
        logits = self.linear(feats)
        return logits.view(B, N, -1)


class GroupModel(nn.Module):
    def __init__(self, in_c: int, hid_c: int = 64):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, hid_c, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool1d(7),
        )
        self.linear = nn.Linear(hid_c * 7, 7, bias=False)

        nn.init.kaiming_normal_(self.conv1d[0].weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.conv1d[0].bias, 0)
        nn.init.kaiming_normal_(self.linear.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, group_vectors):
        B, C, L = group_vectors.size()  # Expected input shape: (B, 1, L)
        x = self.conv1d(group_vectors)
        x = x.view(B, -1)
        logits = self.linear(x)
        return logits.unsqueeze(1)

class Vectorize(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        return x.reshape(x.shape[0], self.channels)
    


class DifferentiablePatchGrouping(nn.Module):
    def __init__(self, num_groups, hidden_dim=256, temperature=0.5):
        """
        Args:
            num_groups: Number of groups to create
            hidden_dim: Hidden dimension for patch embeddings
            temperature: Temperature for softmax (controls assignment sharpness)
        """
        super().__init__()
        self.num_groups = num_groups
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        
        # Enhanced feature extractor with deeper architecture
        self.patch_encoder = nn.Sequential(
            # Initial feature extraction
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Deeper feature extraction
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Global feature pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, hidden_dim)
        )
        
        # Learnable grouping network - this network learns to assign patches to groups
        # without relying on similarity of patch features
        self.grouping_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim*2, num_groups)
        )
        
        # Initialize parameters
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, training=True):
        batch_size, num_patches, channels, patch_h, patch_w = x.shape
        num_patches_per_group = num_patches // self.num_groups
        
        # Extract features from patches
        x_flat = x.reshape(batch_size * num_patches, channels, patch_h, patch_w)
        patch_features = self.patch_encoder(x_flat)
        patch_features = patch_features.view(batch_size, num_patches, self.hidden_dim)
        
        # Learn group assignments directly
        logits = self.grouping_network(patch_features)
        
        # Apply temperature scaling
        logits = logits / self.temperature
        
        if training:
            # Soft assignment for training (differentiable)
            assignment = F.softmax(logits, dim=2)
            
            output = []
            for g in range(self.num_groups):
                # Apply assignment weights to patches
                group_weights = assignment[:, :, g].unsqueeze(2).unsqueeze(3).unsqueeze(4)
                weighted_patches = x * group_weights
                
                # Sort patches by assignment strength
                _, indices = group_weights.squeeze().sort(dim=1, descending=True)
                
                # Select top patches for each batch item
                batch_indices = [weighted_patches[b, indices[b, :num_patches_per_group]] 
                                for b in range(batch_size)]
                group_result = torch.stack(batch_indices)
                output.append(group_result)
            
            output = torch.stack(output, dim=1)
            output = output.view(batch_size, self.num_groups, num_patches_per_group, 
                                channels, patch_h, patch_w)
            
            # Return both output and assignments for loss calculation
            return output, assignment
        else:
            # Hard assignment for inference
            output = torch.zeros(batch_size, self.num_groups, num_patches_per_group, 
                                channels, patch_h, patch_w, device=x.device)
            
            # Get hard assignments
            assignment = F.one_hot(logits.argmax(dim=2), num_classes=self.num_groups)
            
            for b in range(batch_size):
                assigned_patches = torch.zeros(num_patches, dtype=torch.bool, device=x.device)
                
                for g in range(self.num_groups):
                    # Find patches assigned to this group
                    group_indices = torch.where(assignment[b, :, g])[0]
                    
                    # If insufficient patches, fill with unassigned patches
                    if len(group_indices) < num_patches_per_group:
                        remaining = num_patches_per_group - len(group_indices)
                        unassigned = torch.where(~assigned_patches)[0]
                        fill_indices = unassigned[:remaining]
                        group_indices = torch.cat([group_indices, fill_indices])
                    
                    # If too many patches, take the ones with highest logits
                    elif len(group_indices) > num_patches_per_group:
                        group_scores = logits[b, :, g]
                        _, top_indices = group_scores[group_indices].topk(num_patches_per_group)
                        group_indices = group_indices[top_indices]
                    
                    # Add selected patches to this group
                    patches_to_add = x[b, group_indices].view(num_patches_per_group, channels, patch_h, patch_w)
                    output[b, g] = patches_to_add
                    
                    # Mark these patches as assigned
                    assigned_patches[group_indices] = True
            
            return output


# Training function (example)
def train_with_learnable_grouping(model, optimizer, dataloader, downstream_network, num_epochs=10):
    """
    Example training function for the learnable grouping model
    
    Args:
        model: The DifferentiablePatchGrouping model
        optimizer: Optimizer for model parameters
        dataloader: Data loader providing training data
        downstream_network: The network that processes grouped patches
        num_epochs: Number of training epochs
    """
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Forward pass through grouping model
            patches, _ = batch  # Assume batch contains patches and possibly labels
            grouped_patches, assignments = model(patches, training=True)
            
            # Forward pass through downstream network
            downstream_output = downstream_network(grouped_patches)
            
            # Calculate loss using downstream network's output
            loss = downstream_network.loss_function(downstream_output)
            
            # Optional: Add regularization to ensure balanced groups
            # This encourages each group to have a similar number of assigned patches
            balance_loss = balanced_assignment_regularization(assignments)
            total_loss = loss + 0.1 * balance_loss
            
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item()}")


def balanced_assignment_regularization(assignments):
    """
    Encourages balanced assignments across groups
    
    Args:
        assignments: Group assignment probabilities [batch_size, num_patches, num_groups]
    
    Returns:
        Balance loss term
    """
    # Sum assignments across patches for each group
    group_sizes = assignments.sum(dim=1)  # [batch_size, num_groups]
    
    # Ideal size is equal distribution
    batch_size, _, num_groups = assignments.shape
    ideal_size = torch.ones_like(group_sizes) / num_groups
    
    # L1 penalty on deviation from ideal size
    balance_loss = F.l1_loss(group_sizes, ideal_size)
    
    return balance_loss