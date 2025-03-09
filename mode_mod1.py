import os
import torch
import imageio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from losses import TVLoss, perceptual_loss
from dataset_mod1 import *
from srgan_model_mod1 import Generator, Discriminator
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
import logging

from PIL import Image, ImageDraw, ImageFont

# Tracking loss values
pretrain_losses = []
g_losses = []
d_losses = []
epochs_pretrain = []
epochs_finetune = []

# Create a TensorBoard writer
writer = SummaryWriter(log_dir="runs/srgan_experiment")


def check_and_adjust_batch_size(dataset, batch_size):
    num_samples = len(dataset)

    if num_samples % batch_size != 0:
        print(
            f"Warning: The number of samples ({num_samples}) is not divisible by the batch size ({batch_size}). Adjusting the number of samples.")

        # Adjust the number of samples to be divisible by batch size
        adjusted_samples = (num_samples // batch_size) * batch_size
        dataset.LR_img = dataset.LR_img[:adjusted_samples]
        dataset.GT_img = dataset.GT_img[:adjusted_samples]

        print(f"Adjusted number of samples: {adjusted_samples} (now divisible by batch size).")


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

logging.basicConfig(filename="log.txt", level=logging.INFO, format="%(message)s")


def log_training_details(fine_epoch, pre_epoch, patch_size, LR_path, GT_path, fine_tuning):
    log_message = (
        f"Fine-tuned Epoch: {fine_epoch}, Pretrained Epoch: {pre_epoch}, "
        f"Patch Size: {patch_size}, LR Path: {LR_path}, GT Path: {GT_path}, "
        f"Fine Tuning: {fine_tuning}\n"
    )
    logging.info(log_message)  # Log the message to the file



def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([crop(args.scale, args.patch_size), augmentation()])
    dataset = mydata(GT_path=args.GT_path, LR_path=args.LR_path, in_memory=args.in_memory, transform=transform)
    check_and_adjust_batch_size(dataset, args.batch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    generator = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=args.res_num, scale=args.scale).to(device)

    if args.fine_tuning:
        generator.load_state_dict(torch.load(args.generator_path))
        print("Pre-trained model loaded:", args.generator_path)

    generator.train()

    l2_loss = nn.MSELoss()
    g_optim = optim.Adam(generator.parameters(), lr=1e-4)

    pre_epoch = 0
    fine_epoch = 0

    log_training_details(args.fine_train_epoch, args.pre_train_epoch, args.patch_size, args.LR_path, args.GT_path, args.fine_tuning)

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
        writer.add_scalar("Loss/L2_Loss", avg_l2_loss, pre_epoch)  # Log L2 loss
        pretrain_losses.append(avg_l2_loss)
        epochs_pretrain.append(pre_epoch)

        pre_epoch += 1

        if pre_epoch % 50 == 0:
            print(f"Pre-train Epoch {pre_epoch}, Loss: {pretrain_losses[-1]:.6f}")
            plot_loss(epochs_pretrain, pretrain_losses, "L2 Loss", "Loss", "pretrain_L2_loss.png")

        if pre_epoch % 800 == 0:
            torch.save(generator.state_dict(), f'./model_mod1/pre_trained_model_{pre_epoch}.pt')
            # generate(args)

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
        scheduler.step()

        epoch_g_loss = 0
        epoch_d_loss = 0

        for tr_data in tqdm(loader, desc=f"Fine-tuning Epoch {fine_epoch}/{args.fine_train_epoch}"):
            gt = tr_data['GT'].to(device)
            lr = tr_data['LR'].to(device)

            ## **Training Discriminator**
            output, _ = generator(lr)
            fake_prob = discriminator(output)
            real_prob = discriminator(gt)

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

            _percep_loss, hr_feat, sr_feat = VGG_loss((gt + 1.0) / 2.0, (output + 1.0) / 2.0, layer=args.feat_layer)

            l2_loss_value = l2_loss(output, gt)
            percep_loss = args.vgg_rescale_coeff * _percep_loss
            adversarial_loss = args.adv_coeff * cross_ent(fake_prob, real_label)
            total_variance_loss = args.tv_loss_coeff * tv_loss(args.vgg_rescale_coeff * (hr_feat - sr_feat) ** 2)

            g_loss = percep_loss + adversarial_loss + total_variance_loss + l2_loss_value

            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()
            # optim.lr_scheduler.StepLR(g_optim, step_size=2000, gamma=0.1)

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
            torch.save(generator.state_dict(), f'./model_mod1/SRGAN_gene_{fine_epoch}.pt')
            torch.save(discriminator.state_dict(), f'./model_mod1/SRGAN_discrim_{fine_epoch}.pt')


writer.close()


# In[ ]:

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
    dataset = testOnly_data(LR_path=args.LR_path, in_memory=False, transform=None)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    generator = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=args.res_num)
    generator.load_state_dict(torch.load(args.generator_path))
    generator = generator.to(device)
    generator.eval()

    # Directory for original raster metadata and output
    original_raster_dir = r"test_data/delft4"
    output_dir = 'result/delft4'
    os.makedirs(output_dir, exist_ok=True)

    # Get all original raster files dynamically
    raster_files = glob.glob(os.path.join(original_raster_dir, "tile_*.tif"))
    raster_files.sort()  # Sort the files alphabetically (important for consistency)

    # Iterate through the LR tiles and generate SR images
    with torch.no_grad():
        for i, te_data in enumerate(loader):
            # Dynamically match the corresponding original raster
            if i >= len(raster_files):
                print(f"No corresponding raster for index {i}. Skipping...")
                continue

            original_raster = raster_files[i]
            base_name = os.path.basename(original_raster).replace(".tif", "")  # Extract base name without extension
            output_tile_path = os.path.join(output_dir, f"res_{base_name}.tif")

            # Get metadata from the original raster
            with rasterio.open(original_raster) as src:
                transform = src.transform  # Get the original affine transform
                crs = src.crs  # Get the CRS
                profile = src.profile  # Get raster profile

                # Generate SR output
                lr = te_data['LR'].to(device)
                output, _ = generator(lr)
                output = output[0].cpu().numpy()
                output = (output + 1.0) / 2.0  # Rescale to [0, 1]
                output = output.transpose(1, 2, 0)  # Rearrange dimensions for saving

                # Update profile to match the SR resolution and output data
                profile.update({
                    "height": output.shape[0],
                    "width": output.shape[1],
                    "transform": Affine(
                        transform.a / args.scale, transform.b, transform.c,
                        transform.d, transform.e / args.scale, transform.f
                    ),  # Adjust affine for scaling
                    "dtype": "uint8",
                    "count": output.shape[2],  # Number of bands (e.g., RGB = 3)
                })

                # Write SR output as a GeoTIFF
                with rasterio.open(output_tile_path, "w", **profile) as dst:
                    dst.write((output * 255).astype(np.uint8).transpose(2, 0, 1))  # Save as uint8

            print(f"Saved SR image with metadata: {output_tile_path}")

def generate(args, model_dir='model_mod1', output_dir='generated_photos_mod1', gif_output_dir='progress_gifs_mod1', samples=10, model_type="pre_trained"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = testOnly_data(LR_path=args.LR_path, in_memory=False, transform=None)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gif_output_dir, exist_ok=True)

    # Get all models in the model directory
    if model_type == "pre_trained":
        model_files = sorted(glob.glob(os.path.join(model_dir, "pre_trained_model_*.pt")))
    elif model_type == "SRGAN_gene":
        model_files = sorted(glob.glob(os.path.join(model_dir, "SRGAN_gene_*.pt")))
    else:
        print("Invalid model type. Use 'pre_trained' or 'SRGAN_gene'.")
        return

    # Prepare dictionary to store images for each tile
    image_paths = {f"tile_{i}": [] for i in range(samples)}  # Store paths for each tile across epochs

    # Process images for each model (epoch)
    for model_file in model_files:
        epoch = int(model_file.split('_')[-1].replace('.pt', ''))  # Extract epoch from model filename
        print(f"Generating images for model at epoch {epoch}")

        # Load the generator model
        generator = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=args.res_num)
        generator.load_state_dict(torch.load(model_file))
        generator = generator.to(device)
        generator.eval()

        # Create a directory for the generated images of this epoch
        epoch_output_dir = os.path.join(output_dir, f"epoch{epoch}")
        os.makedirs(epoch_output_dir, exist_ok=True)

        with torch.no_grad():
            for i, te_data in enumerate(loader):
                if i >= samples:  # Only process the first 10 images
                    break

                lr = te_data['LR'].to(device)
                output, _ = generator(lr)
                output = output[0].cpu().numpy()
                output = (output + 1.0) / 2.0  # Rescale to [0, 1]
                output = output.transpose(1, 2, 0)  # Rearrange dimensions for saving

                # Generate filename based on tile name and epoch
                base_name = f"tile_{i}_epoch{epoch}"
                output_image_path = os.path.join(epoch_output_dir, f'{base_name}.png')

                # Add the epoch number on the image
                image = Image.fromarray((output * 255).astype(np.uint8))
                # image = image.convert("RGB")  # Convert to RGB for drawing text
                draw = ImageDraw.Draw(image)
                font = ImageFont.load_default(size=40)  # Use default font or specify a TTF font
                text = f"Epoch: {epoch}"
                draw.text((10, 10), text, font=font, fill=(255, 0, 0))  # Red color for the text
                image.save(output_image_path)

                print(f"Saved generated image for epoch {epoch} to {output_image_path}")

                # Store image paths by tile index for GIF creation
                image_paths[f"tile_{i}"].append(output_image_path)

    # Now create a GIF for each tile by collecting its images across different epochs
    for tile, paths in image_paths.items():
        paths.sort(key=lambda x: int(x.split('_')[-1].replace('epoch', ' ').replace('.png', '')))
        gif_path = os.path.join(gif_output_dir, f'{tile}_progress.gif')
        with imageio.get_writer(gif_path, mode='I', duration=1000, loop = 0) as writer:  # Adjust duration to 1 second
            for image_path in paths:
                image = Image.open(image_path)
                writer.append_data(np.array(image))  # Append image as numpy array

        print(f"GIF for {tile} saved to {gif_path}")
