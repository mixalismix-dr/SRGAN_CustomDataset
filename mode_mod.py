import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from losses import TVLoss, perceptual_loss
from dataset_mod import *
from srgan_model_mod import Generator, Discriminator
from vgg19_model import vgg19
import numpy as np
from tqdm import tqdm
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio
import rasterio
import glob
from rasterio.transform import Affine
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import traceback

# Tracking loss values
pretrain_losses = []
g_losses = []
d_losses = []
epochs_pretrain = []
epochs_finetune = []

# Create a TensorBoard writer
writer = SummaryWriter(log_dir="runs/srgan_experiment")

def plot_loss(epochs, losses, primary_label, ylabel, filename, second_losses=None, second_label=None):
    os.makedirs("loss_plots", exist_ok=True)
    plt.figure(figsize=(10, 5))

    # Plot primary loss (L2 loss OR Generator Loss)
    plt.plot(epochs, losses, label=primary_label, color='blue' if "L2" in primary_label else 'red')

    # If a second loss (Discriminator Loss) is provided, plot it on the same graph
    if second_losses is not None and second_label is not None:
        plt.plot(epochs, second_losses, label=second_label, color='green')

    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(f"{primary_label} Loss Curve" if second_losses is None else "Generator & Discriminator Losses")
    plt.legend()
    plt.savefig(f"loss_plots/{filename}")
    plt.close()

def train(args):

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transform = transforms.Compose([crop(args.scale, args.patch_size), augmentation()])
        dataset = mydata(GT_path=args.GT_path, LR_path=args.LR_path, mask_path=args.mask_path, in_memory=args.in_memory, transform=transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        generator = Generator(img_feat=4, n_feats=64, kernel_size=3, num_block=args.res_num, scale=args.scale).to(device)

        if args.fine_tuning:
            generator.load_state_dict(torch.load(args.generator_path))
            print("Pre-trained model loaded:", args.generator_path)

        generator.train()

        l2_loss = nn.MSELoss()
        g_optim = optim.Adam(generator.parameters(), lr=1e-4)

        pre_epoch = 0
        fine_epoch = 0

        #### **Pre-Training Using L2 Loss**
        while pre_epoch < args.pre_train_epoch:
            epoch_loss = 0
            for tr_data in tqdm(loader, desc=f"Pre-training Epoch {pre_epoch}/{args.pre_train_epoch}"):
                gt = tr_data['GT'].to(device)
                lr = tr_data['LR'].to(device)

                output, _ = generator(lr)
                loss = l2_loss(gt, output)

                g_optim.zero_grad()
                loss.backward()
                g_optim.step()

                epoch_loss += loss.item()

            optim.lr_scheduler.StepLR(g_optim, step_size=2000, gamma=0.1)

            pretrain_losses.append(epoch_loss / len(loader))
            epochs_pretrain.append(pre_epoch)
            avg_l2_loss = epoch_loss / len(loader)
            writer.add_scalar("Loss/L2_Loss", avg_l2_loss, pre_epoch)
            pretrain_losses.append(avg_l2_loss)
            epochs_pretrain.append(pre_epoch)

            pre_epoch += 1

            if pre_epoch % 50 == 0:
                print(f"Pre-train Epoch {pre_epoch}, Loss: {pretrain_losses[-1]:.6f}")
                plot_loss(epochs_pretrain, pretrain_losses, "L2 Loss", "Loss", "pretrain_L2_loss.png")

            if pre_epoch % 800 == 0:
                torch.save(generator.state_dict(), f'./model/pre_trained_model_{pre_epoch}.pt')

        #### **Fine-Tuning Using Perceptual & Adversarial Loss**
        vgg_net = vgg19().to(device).eval()
        discriminator = Discriminator(patch_size=args.patch_size * args.scale).to(device).train()

        d_optim = optim.Adam(discriminator.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(g_optim, step_size=2000, gamma=0.1)

        VGG_loss = perceptual_loss(vgg_net)
        cross_ent = nn.BCELoss()
        tv_loss = TVLoss()

        real_label = torch.ones((args.batch_size, 1)).to(device)
        fake_label = torch.zeros((args.batch_size, 1)).to(device)

        while fine_epoch < args.fine_train_epoch:

            epoch_g_loss = 0
            epoch_d_loss = 0

            for tr_data in tqdm(loader, desc=f"Fine-tuning Epoch {fine_epoch}/{args.fine_train_epoch}"):
                gt = tr_data['GT'].to(device)
                lr = tr_data['LR'].to(device)

                ## **Training Discriminator**
                # For the generator (train with both lr and mask_lr):
                output, _ = generator(lr)  # Pass to the generator
                fake_prob = discriminator(output)
                real_prob = discriminator(gt)

                real_label = torch.ones_like(real_prob).to(device)
                fake_label = torch.zeros_like(fake_prob).to(device)

                d_loss_real = cross_ent(real_prob, real_label)
                d_loss_fake = cross_ent(fake_prob, fake_label)
                d_loss = d_loss_real + d_loss_fake

                d_optim.zero_grad()
                d_loss.backward()
                d_optim.step()
                # optim.lr_scheduler.StepLR(d_optim, step_size=2000, gamma=0.1)

                ## **Training Generator**
                output, _ = generator(lr)
                fake_prob = discriminator(output)

                # Slice only the RGB part for VGG perceptual loss
                gt_rgb = (gt[:, :3, :, :] + 1.0) / 2.0
                output_rgb = (output[:, :3, :, :] + 1.0) / 2.0

                _percep_loss, hr_feat, sr_feat = VGG_loss(gt_rgb, output_rgb, layer=args.feat_layer)

                l2_loss_value = l2_loss(output, gt)
                percep_loss = args.vgg_rescale_coeff * _percep_loss
                adversarial_loss = args.adv_coeff * cross_ent(fake_prob, real_label)
                total_variance_loss = args.tv_loss_coeff * tv_loss(args.vgg_rescale_coeff * (hr_feat - sr_feat) ** 2)

                g_loss = percep_loss + adversarial_loss + total_variance_loss + l2_loss_value

                g_optim.zero_grad()
                g_loss.backward()
                g_optim.step()
                # optim.lr_scheduler.StepLR(g_optim, step_size=2000, gamma=0.1)
                scheduler.step()

                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()

            optim.lr_scheduler.StepLR(g_optim, step_size=2000, gamma=0.1)

            avg_g_loss = epoch_g_loss / len(loader)
            avg_d_loss = epoch_d_loss / len(loader)

            writer.add_scalar("Loss/Generator_Loss", avg_g_loss, fine_epoch)  # Log Generator loss
            writer.add_scalar("Loss/Discriminator_Loss", avg_d_loss, fine_epoch)  # Log Discriminator loss

            g_losses.append(epoch_g_loss / len(loader))
            d_losses.append(epoch_d_loss / len(loader))
            epochs_finetune.append(fine_epoch)
            fine_epoch += 1

            if fine_epoch % 50 == 0:
                print(f"Fine-tune Epoch {fine_epoch}, G Loss: {g_losses[-1]:.6f}, D Loss: {d_losses[-1]:.6f}")
                plot_loss(epochs_finetune, g_losses, "Generator Loss", "Loss", "gan_losses.png",
                          second_losses=d_losses, second_label="Discriminator Loss")

            if fine_epoch % 500 == 0:
                torch.save(generator.state_dict(), f'./model/SRGAN_gene_{fine_epoch}.pt')
                torch.save(discriminator.state_dict(), f'./model/SRGAN_discrim_{fine_epoch}.pt')

        writer.close()


    except Exception as e:

        with open("error_log.txt", "a") as f:

            f.write(f"\nError during training at fine_epoch {fine_epoch}:\n")

            f.write(traceback.format_exc())

        print(f"An error occurred. Check error_log.txt for details.")

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    dataset = mydata(GT_path=args.GT_path, LR_path=args.LR_path, in_memory=False, transform=None)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    generator = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=args.res_num)
    generator.load_state_dict(torch.load(args.generator_path))
    generator = generator.to(device)
    generator.eval()

    f = open('./result1.txt', 'w')
    psnr_list = []

    with torch.no_grad():
        for i, te_data in enumerate(loader):
            gt = te_data['GT'].to(device)
            lr = te_data['LR'].to(device)

            bs, c, h, w = lr.size()
            gt = gt[:, :, : h * args.scale, : w * args.scale]

            output, _ = generator(lr)

            output = output[0].cpu().numpy()
            output = np.clip(output, -1.0, 1.0)
            gt = gt[0].cpu().numpy()

            output = (output + 1.0) / 2.0
            gt = (gt + 1.0) / 2.0

            output = output.transpose(1, 2, 0)
            gt = gt.transpose(1, 2, 0)

            y_output = rgb2ycbcr(output)[args.scale:-args.scale, args.scale:-args.scale, :1]
            y_gt = rgb2ycbcr(gt)[args.scale:-args.scale, args.scale:-args.scale, :1]

            psnr = peak_signal_noise_ratio(y_output / 255.0, y_gt / 255.0, data_range=1.0)
            psnr_list.append(psnr)
            f.write('psnr : %04f \n' % psnr)

            result = Image.fromarray((output * 255.0).astype(np.uint8))
            result.save('./result/res_%04d.tif' % i)

        f.write('avg psnr : %04f' % np.mean(psnr_list))

def test_only(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = testOnly_data(
        LR_path=args.LR_path,
        mask_path=r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\other scripts\output_mask_tiles",
        in_memory=False,
        transform=None
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    generator = Generator(img_feat=4, n_feats=64, kernel_size=3, num_block=args.res_num)
    generator.load_state_dict(torch.load(args.generator_path))
    generator = generator.to(device)
    generator.eval()

    original_raster_dir = r"test_data/delft4_edge"
    output_dir = 'result/delft4'
    os.makedirs(output_dir, exist_ok=True)

    raster_files = glob.glob(os.path.join(original_raster_dir, "tile_*.tif"))
    raster_files.sort()

    with torch.no_grad():
        for i, te_data in enumerate(loader):
            if i >= len(raster_files):
                print(f"No corresponding raster for index {i}. Skipping...")
                continue

            original_raster = raster_files[i]
            base_name = os.path.basename(original_raster).replace(".tif", "")
            output_tile_path = os.path.join(output_dir, f"res_{base_name}.tif")
            mask_tile_path = os.path.join(output_dir, f"mask_{base_name}.tif")

            with rasterio.open(original_raster) as src:
                transform = src.transform
                crs = src.crs
                profile = src.profile

                lr = te_data['LR'].to(device)
                mask = te_data['mask_lr'].to(device)
                mask_lr = mask.cpu().numpy()[0, 0]
                mask_lr = Image.fromarray(mask_lr).resize((64, 64), Image.BILINEAR)
                mask_lr = np.expand_dims(np.array(mask_lr), axis=0)
                mask_lr = torch.tensor(mask_lr).to(device).unsqueeze(0)

                input_data = torch.cat((lr, mask_lr), dim=1)  # Concatenate LR and mask for generator input
                output, _ = generator(input_data)  # Pass to the generator

                output = output[0].cpu().numpy()
                output = (output + 1.0) / 2.0
                output = output.transpose(1, 2, 0)

                output_rgb = output[:, :, :3]  # RGB bands
                output_mask = output[:, :, 3]  # Mask band

                # Save RGB image
                rgb_profile = profile.copy()
                rgb_profile.update({
                    "height": output.shape[0],
                    "width": output.shape[1],
                    "transform": Affine(
                        transform.a / args.scale, transform.b, transform.c,
                        transform.d, transform.e / args.scale, transform.f
                    ),
                    "dtype": "uint8",
                    "count": 3,
                })

                with rasterio.open(output_tile_path, "w", **rgb_profile) as dst:
                    dst.write((output_rgb * 255).astype(np.uint8).transpose(2, 0, 1))

                # Save mask image
                mask_profile = profile.copy()
                mask_profile.update({
                    "height": output.shape[0],
                    "width": output.shape[1],
                    "transform": Affine(
                        transform.a / args.scale, transform.b, transform.c,
                        transform.d, transform.e / args.scale, transform.f
                    ),
                    "dtype": "uint8",
                    "count": 1,
                })

                with rasterio.open(mask_tile_path, "w", **mask_profile) as dst:
                    dst.write((output_mask * 255).astype(np.uint8), 1)

            print(f"Saved SR image: {output_tile_path}")
            print(f"Saved SR mask: {mask_tile_path}")
