import os, sys
import logging
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import PatchModel, PatchGroupingNet, ConvModel, DifferentiablePatchGrouping
from dataset_builder import build_dataset
from puzzle_utils import batch_tch_divide_image, reconstruct_image_from_groups, get_groups, model_based_grouping
import numpy as np
from utils import gumbel_sinkhorn_ops, metric


def stack_patches_in_group(x):
    B, G, N, C, H, W = x.shape
    assert N == 4, "This function assumes stacking 4 patches (2x2 grid)"

    # Reshape to (B*G, 4, C, H, W)
    x = x.view(B * G, 4, C, H, W)

    # Arrange the 4 patches into a 2x2 grid
    top = torch.cat([x[:, 0], x[:, 1]], dim=3)  # (B*G, C, H, 2*W)
    bottom = torch.cat([x[:, 2], x[:, 3]], dim=3)
    stacked = torch.cat([top, bottom], dim=2)  # (B*G, C, 2*H, 2*W)

    return stacked.view(B, G, C, 2 * H, 2 * W)

def unstack_patches_in_group(x):
    B, G, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0, "H and W must be even"

    h, w = H // 2, W // 2
    x = x.view(B * G, C, H, W)

    # Extract the 4 patches
    patch_0 = x[:, :, :h, :w]  # top-left
    patch_1 = x[:, :, :h, w:]  # top-right
    patch_2 = x[:, :, h:, :w]  # bottom-left
    patch_3 = x[:, :, h:, w:]  # bottom-right

    unstacked = torch.stack([patch_0, patch_1, patch_2, patch_3], dim=1)  # (B*G, 4, C, h, w)
    return unstacked.view(B, G, 4, C, h, w)

def train(cfg):
    logger = logging.getLogger("JigsawPuzzle")
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"

    if cfg.dataset == "MNIST":
        in_c = 1
    else:
        in_c = 3

    model_patch = ConvModel(in_c, cfg.pieces//cfg.groups, cfg.image_size//cfg.groups, cfg.hid_c, cfg.stride, cfg.kernel_size).to(device)
    model_group = ConvModel(in_c, cfg.groups, cfg.image_size, cfg.hid_c, cfg.stride, cfg.kernel_size).to(device)

    model_grouping = DifferentiablePatchGrouping(num_groups=cfg.groups**2)

    optimizer = torch.optim.Adam(list(model_patch.parameters()) + list(model_group.parameters()), lr=cfg.lr)

    train_data = build_dataset(cfg, split="train")
    train_loader = DataLoader(train_data, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)

    model_patch.train()
    model_group.train()
    model_grouping.train()
    
    # Calculate patch size based on configuration
    patch_size = cfg.image_size // cfg.pieces

    for epoch in range(cfg.epochs):
        sum_loss = 0
        for i, data in enumerate(train_loader):
            inputs, _ = data
            if cfg.image_size > 28:
                inputs = F.interpolate(inputs, size=(cfg.image_size, cfg.image_size), mode='bilinear', align_corners=False)
            pieces, random_pieces, _ = batch_tch_divide_image(inputs, cfg.pieces)
            pieces, random_pieces = pieces.to(device), random_pieces.to(device)

            batch_size = inputs.size(0)

            random_group_pieces, _ = model_grouping(random_pieces, training=True)
            #patch_groups_indices = model_based_grouping(random_pieces.view(-1, *random_pieces.shape[2:]), batch_size, num_groups=cfg.groups**2, model=model_grouping)
            #patch_groups_indices = get_groups(random_pieces.view(-1, *random_pieces.shape[2:]), batch_size, num_groups=cfg.groups**2)
            group_ordered_patches = []

            # Store patch permutations for Kendall Tau calculation
            patch_order_list = []

            for g in range(cfg.groups**2):
                # group_indices = patch_groups_indices[:, g, :].reshape(-1)
                # group_random_patches = random_pieces.view(-1, *random_pieces.shape[2:])[group_indices]
                # group_random_patches = group_random_patches.view(batch_size, -1, *group_random_patches.shape[1:])
                group_random_patches = random_group_pieces[:, g, :]

                log_alpha_patch = model_patch(group_random_patches)
                patch_order_list.append(log_alpha_patch)
                gumbel_sinkhorn_mat = gumbel_sinkhorn_ops.gumbel_sinkhorn(log_alpha_patch, cfg.tau, cfg.n_sink_iter)
                group_ordered_pieces = gumbel_sinkhorn_ops.inverse_permutation_for_image(group_random_patches, gumbel_sinkhorn_mat)
                group_ordered_patches.append(group_ordered_pieces)

            group_vectors_tensor = torch.stack(group_ordered_patches, dim=1)  # (B, G, D, 1)
            random_groups = stack_patches_in_group(group_vectors_tensor)
            log_alpha_group = model_group(random_groups)
            gumbel_sinkhorn_mat = gumbel_sinkhorn_ops.gumbel_sinkhorn(log_alpha_group, cfg.tau, cfg.n_sink_iter)
            ordered_groups = gumbel_sinkhorn_ops.inverse_permutation_for_image(random_groups, gumbel_sinkhorn_mat)
            #ordered_group_vectors_tensor = unstack_patches_in_group(ordered_groups)
            full_imgs = reconstruct_image_from_groups(ordered_groups)

            loss = torch.nn.functional.mse_loss(full_imgs, inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()

            if cfg.display > 0 and ((i+1) % cfg.display) == 0:
                logger.info("epoch %i [%i/%i] loss %f", epoch, i+1, len(train_loader), loss.item())
        logger.info("epoch %i|  mean loss %f", epoch, sum_loss/len(train_loader))

        torch.save(model_patch.state_dict(), os.path.join(cfg.out_dir, "model_patch.pth"))
        torch.save(model_group.state_dict(), os.path.join(cfg.out_dir, "model_group.pth"))
        torch.save(model_grouping.state_dict(), os.path.join(cfg.out_dir, "model_grouping.pth"))
    
    return {"final_loss": sum_loss/len(train_loader)}

def evaluation(cfg):
    logger = logging.getLogger("JigsawPuzzle")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    in_c = 1 if cfg.dataset == "MNIST" else 3

    # Load models
    model_patch = ConvModel(in_c, cfg.pieces // cfg.groups, cfg.image_size // cfg.groups, cfg.hid_c, cfg.stride, cfg.kernel_size)
    model_group = ConvModel(in_c, cfg.groups, cfg.image_size, cfg.hid_c, cfg.stride, cfg.kernel_size)
    model_grouping = DifferentiablePatchGrouping(num_groups=cfg.groups**2)
    model_patch.load_state_dict(torch.load(os.path.join(cfg.out_dir, "model_patch.pth")))
    model_group.load_state_dict(torch.load(os.path.join(cfg.out_dir, "model_group.pth")))
    model_grouping.load_state_dict(torch.load(os.path.join(cfg.out_dir, "model_grouping.pth")))
    model_patch = model_patch.to(device).eval()
    model_group = model_group.to(device).eval()
    model_grouping = model_grouping.to(device).eval()

    # Load data
    eval_data = build_dataset(cfg, split="test")
    loader = DataLoader(eval_data, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=False)
    
    # Calculate patch size based on configuration
    patch_size = cfg.image_size // cfg.pieces

    # Metrics
    l1_diffs = []
    l2_diffs = []
    prop_wrongs = []
    prop_any_wrongs = []
    kendall_taus = []

    with torch.no_grad():
        logger.info("start evaluation")
        for i, data in enumerate(loader):
            inputs, _ = data
            if cfg.image_size > 28:
                inputs = F.interpolate(inputs, size=(cfg.image_size, cfg.image_size), mode='bilinear', align_corners=False)
            pieces, random_pieces, perm_index = batch_tch_divide_image(inputs, cfg.pieces)
            pieces, random_pieces = pieces.to(device), random_pieces.to(device)

            batch_size = inputs.size(0)

            random_group_pieces = model_grouping(random_pieces, training=False)
            #patch_groups_indices = model_based_grouping(random_pieces.view(-1, *random_pieces.shape[2:]), batch_size, num_groups=cfg.groups**2, model=model_grouping)
            #patch_groups_indices = get_groups(random_pieces.view(-1, *random_pieces.shape[2:]), batch_size, num_groups=cfg.groups**2)
            group_ordered_patches = []

            # Store patch permutations for Kendall Tau calculation
            patch_order_list = []

            for g in range(cfg.groups**2):
                # group_indices = patch_groups_indices[:, g, :].reshape(-1)
                # group_random_patches = random_pieces.view(-1, *random_pieces.shape[2:])[group_indices]
                # group_random_patches = group_random_patches.view(batch_size, -1, *group_random_patches.shape[1:])
                group_random_patches = random_group_pieces[:, g, :]

                log_alpha_patch = model_patch(group_random_patches)
                patch_order_list.append(log_alpha_patch)
                gumbel_sinkhorn_mat = gumbel_sinkhorn_ops.gumbel_matching(log_alpha_patch, noise=False)
                group_ordered_pieces = gumbel_sinkhorn_ops.inverse_permutation_for_image(group_random_patches, gumbel_sinkhorn_mat)
                group_ordered_patches.append(group_ordered_pieces)

            group_vectors_tensor = torch.stack(group_ordered_patches, dim=1)  # (B, G, D, 1)
            random_groups = stack_patches_in_group(group_vectors_tensor)
            log_alpha_group = model_group(random_groups)
            gumbel_sinkhorn_mat = gumbel_sinkhorn_ops.gumbel_matching(log_alpha_group, noise=False)
            ordered_groups = gumbel_sinkhorn_ops.inverse_permutation_for_image(random_groups, gumbel_sinkhorn_mat)
            #ordered_group_vectors_tensor = unstack_patches_in_group(ordered_groups)
            full_imgs = reconstruct_image_from_groups(ordered_groups)
                        
            # Reconstruct full images for visualization if needed
            est_ordered_pieces, _, _ = batch_tch_divide_image(full_imgs, cfg.pieces)

            # === Metric calculations ===
            hard_l1_diff = (pieces - est_ordered_pieces).abs().mean((2,3,4))  # (B, N)
            hard_l2_diff = (pieces - est_ordered_pieces).pow(2).mean((2,3,4))
            sign_l1_diff = hard_l1_diff.sign()
            prop_wrong = sign_l1_diff.mean(1)
            prop_any_wrong = sign_l1_diff.sum(1).sign()

            # Calculate Kendall Tau if permutation index is available
            if perm_index is not None:
                est_patch_perm = torch.cat([p.max(1)[1] for p in patch_order_list], dim=1).float()
                gt_patch_perm = perm_index.to("cpu").float()
                np_est_perm = est_patch_perm.cpu().numpy()
                np_gt_perm = gt_patch_perm.cpu().numpy()
                kendall = metric.kendall_tau(np_est_perm, np_gt_perm)
                kendall_taus.append(kendall)
            else:
                # Use zeros as placeholder if permutation index is not available
                kendall_taus.append(np.zeros(batch_size))

            l1_diffs.append(hard_l1_diff.cpu())
            l2_diffs.append(hard_l2_diff.cpu())
            prop_wrongs.append(prop_wrong.cpu())
            prop_any_wrongs.append(prop_any_wrong.cpu())

        # === Aggregate ===
        mean_l1_diff = torch.cat(l1_diffs).mean()
        mean_l2_diff = torch.cat(l2_diffs).mean()
        mean_prop_wrong = torch.cat(prop_wrongs).mean()
        mean_prop_any_wrong = torch.cat(prop_any_wrongs).mean()
        mean_kendall_tau = np.concatenate(kendall_taus).mean()

        logger.info("\nmean l1 diff : %f\n mean l2 diff : %f\n mean prop wrong : %f\n mean prop any wrong : %f\n mean kendall tau : %f",
            mean_l1_diff, mean_l2_diff, mean_prop_wrong, mean_prop_any_wrong, mean_kendall_tau
        )

        return {
            "l1": mean_l1_diff.item(),
            "l2": mean_l2_diff.item(),
            "prop_wrong": mean_prop_wrong.item(),
            "prop_any_wrong": mean_prop_any_wrong.item(),
            "kendall_tau": mean_kendall_tau
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # directory option
    parser.add_argument("--root", "-r", default="./data", type=str, help="dataset root directory")
    parser.add_argument("--out_dir", "-o", default="./log", type=str, help="output directory")
    # optimizer option
    parser.add_argument("--epochs", "-e", default=4, type=int, help="number of epochs")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--batch_size", default=16, type=int, help="mini-batch size")
    parser.add_argument("--num_workers", default=8, type=int, help="number of threads for CPU parallel")
    # dataset option
    parser.add_argument("--dataset", default="MNIST", type=str, help="dataset name chosen from ['MNIST',]")
    parser.add_argument("--pieces", "-p", default=8, type=int, help="number of pieces each side")
    parser.add_argument("--groups", "-g", default=4, type=int, help="number of groups each side")
    parser.add_argument("--image_size", default=32, type=int, help="original image size")
    # model parameter option
    parser.add_argument("--hid_c", default=64, type=int, help="number of hidden channels")
    parser.add_argument("--stride", default=1, type=int, help="stride in pooling operator")
    parser.add_argument("--kernel_size", default=5, type=int, help="kernel size in convolution operator")
    # Gumbel sinkhorn option
    parser.add_argument("--tau", default=1.0, type=float, help="temperture parameter")
    parser.add_argument("--n_sink_iter", default=20, type=int, help="number of iterations for sinkhorn normalization")
    parser.add_argument("--n_samples", default=5, type=int, help="number of samples from gumbel-sinkhorn distribution")
    # misc option
    parser.add_argument("--display", default=50, type=int, help="display loss every 'display' iteration. if set to 0, won't display")
    parser.add_argument("--eval_only", action="store_true", help="evaluation without training")

    cfg = parser.parse_args()

    if not os.path.exists(cfg.out_dir):
        os.makedirs(cfg.out_dir, exist_ok=True)  # Use makedirs with exist_ok flag for better error handling

    # logger setup
    logging.basicConfig(
        filename=os.path.join(cfg.out_dir, "console.log"),
    )
    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    logger = logging.getLogger("JigsawPuzzle")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = plain_formatter
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if not cfg.eval_only:
        pass
        train_results = train(cfg)
    
    eval_results = evaluation(cfg)