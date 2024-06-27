import dis
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets

import torch.nn as nn
import torch.nn.functional as functional
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from PIL import Image
from IPython.display import display
import cv2
import random

import os
import numpy as np
import scipy.sparse
import pyamg

from Discriminator128 import Discriminator
from Generator128 import Generator

fid = FrechetInceptionDistance(feature=64)

SIZE = 128

transform = transforms.Compose(
    [
        # resize the image to SIZExSIZE
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

NUM_ITERATIONS = 300
MODELS_PATH = "models\\rms_slow"


# pre-process the mask array so that uint64 types from opencv.imread can be adapted
def prepare_mask(mask):
    if type(mask[0][0]) is np.ndarray:
        result = np.ndarray((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if sum(mask[i][j]) > 0:
                    result[i][j] = 1
                else:
                    result[i][j] = 0
        mask = result
    return mask


def blend(img_target, img_source, img_mask, offset=(0, 0)):  # offset=(40, -30)
    # compute regions to be blended
    region_source = (
        max(-offset[0], 0),
        max(-offset[1], 0),
        min(img_target.shape[0] - offset[0], img_source.shape[0]),
        min(img_target.shape[1] - offset[1], img_source.shape[1]),
    )
    region_target = (
        max(offset[0], 0),
        max(offset[1], 0),
        min(img_target.shape[0], img_source.shape[0] + offset[0]),
        min(img_target.shape[1], img_source.shape[1] + offset[1]),
    )
    region_size = (
        region_source[2] - region_source[0],
        region_source[3] - region_source[1],
    )

    # clip and normalize mask image
    img_mask = img_mask[
        region_source[0] : region_source[2], region_source[1] : region_source[3]
    ]
    img_mask = prepare_mask(img_mask)
    img_mask[img_mask == 0] = False
    # img_mask[img_mask != False] = True
    img_mask[img_mask != 0] = True

    # create coefficient matrix
    # a_ = scipy.sparse.identity(np.prod(region_size), format='lil')
    a_ = scipy.sparse.identity(int(np.prod(region_size)), format="lil")
    for y in range(region_size[0]):
        for x in range(region_size[1]):
            if img_mask[y, x]:
                index = x + y * region_size[1]
                a_[index, index] = 4
                if index + 1 < np.prod(region_size):
                    a_[index, index + 1] = -1
                if index - 1 >= 0:
                    a_[index, index - 1] = -1
                if index + region_size[1] < np.prod(region_size):
                    a_[index, index + region_size[1]] = -1
                if index - region_size[1] >= 0:
                    a_[index, index - region_size[1]] = -1
    a_ = a_.tocsr()

    # create poisson matrix for b
    p_ = pyamg.gallery.poisson(img_mask.shape)

    # for each layer (ex. RGB)
    for num_layer in range(img_target.shape[2]):
        # get subimages
        t = img_target[
            region_target[0] : region_target[2],
            region_target[1] : region_target[3],
            num_layer,
        ]
        s = img_source[
            region_source[0] : region_source[2],
            region_source[1] : region_source[3],
            num_layer,
        ]
        t = t.flatten()
        s = s.flatten()

        # create b
        b = p_ * s
        for y in range(region_size[0]):
            for x in range(region_size[1]):
                if not img_mask[y, x]:
                    index = x + y * region_size[1]
                    b[index] = t[index]

        # solve Ax = b
        x = pyamg.solve(a_, b, verb=False, tol=1e-10)

        # assign x to target image
        x = np.reshape(x, region_size)
        x[x > 255] = 255
        x[x < 0] = 0
        x = np.array(x, img_target.dtype)
        img_target[
            region_target[0] : region_target[2],
            region_target[1] : region_target[3],
            num_layer,
        ] = x

    return img_target


def blend_all(target_img_path, source_img_path, mask_path, index=0):
    target = cv2.imread(target_img_path)
    source = cv2.imread(source_img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    cv2.dilate(mask, np.ones((5, 5), np.uint8), mask, iterations=1)

    print(np.max(target), np.min(target))

    blend_img = blend(target, source, mask)
    print(np.max(blend_img), np.min(blend_img))
    cv2.imwrite(f"working\\output-{index}.jpg", blend_img)


def create_mask(x_0, y_0, x_1, y_1, size=(SIZE, SIZE)):
    arr = np.zeros(size)

    arr[x_0:x_1, y_0:y_1] = 1
    return arr


def get_ring(mask):
    ring = np.zeros_like(mask)
    kernel = np.array([[0.01, 1, 0.01], [0.01, 1, 0.01], [0.01, 1, 0.01]])
    for k in range(1, mask.shape[0] - 1):
        if k == SIZE // 2:
            k = SIZE // 2

        new_arr = np.zeros_like(mask)
        for i in range(1, mask.shape[0] - 1):
            for j in range(1, mask.shape[1] - 1):
                new_arr[i, j] = min(
                    np.sum(mask[i - 1 : i + 2, j - 1 : j + 2] * kernel), 1
                )
        ring += (new_arr - mask) / (2**(k//2))
        # ring += (new_arr - mask) / (k**2)
        mask = new_arr

    # for i in range(SIZE):
    #     for j in range(SIZE):
    #         if j < 0.4 * SIZE or j > 0.6 * SIZE:
    #             ring[i, j] /= 2

    return ring * 2


def get_images(images_path):

    dataset = datasets.ImageFolder(images_path, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    # load image
    for i, (real_images, _) in enumerate(data_loader):
        img = real_images
        yield img


def solve(images_path):

    # load models

    generator = Generator()
    discriminator = Discriminator()
    discriminator70 = Discriminator()

    generator.load_state_dict(
        torch.load(
            os.path.join(MODELS_PATH, "generator"), map_location=torch.device("cpu")
        )
    )
    discriminator.load_state_dict(
        torch.load(
            os.path.join(MODELS_PATH, "discriminator"), map_location=torch.device("cpu")
        )
    )
    discriminator.load_state_dict(
        torch.load(
            os.path.join(MODELS_PATH, "discriminator_70"), map_location=torch.device("cpu")
        )
    )

    # Define the loss function
    criterion = nn.L1Loss()
    bce_criterion = nn.BCELoss()

    org_mask = create_mask(
        int(40 * SIZE / 64),
        int(24 * SIZE / 64),
        int(50 * SIZE / 64),
        int(40 * SIZE / 64),
    )
    # Create a 64x64 tensor filled with False
    mask = np.array([get_ring(org_mask), get_ring(org_mask), get_ring(org_mask)])
    cv2.imwrite("working\\mask.jpg", mask[0] * 255)
    mask = torch.from_numpy(mask).unsqueeze(0)

    org_mask_cuda = torch.from_numpy(org_mask).unsqueeze(0)

    for img_num, img in enumerate(get_images(images_path)):

        masked_img = img * mask
        original_blacked_img = img.cpu() * (1 - org_mask)
        original_blacked_img = original_blacked_img

        # show initial image
        save_image(img, f"working\\temp.png", nrow=5, normalize=True)
        display_img = Image.open(f"working\\temp.png")
        display(display_img)

        # Initialize the latent vector z with random noise
        noise = (
            2 * torch.rand((1, Generator.LATENT_DIM, 1, 1), dtype=torch.float32)
        ) - 1
        noise = noise
        noise.requires_grad = True  # Ensure gradients are tracked

        # Set up the optimizer for z
        optimizer = torch.optim.RMSprop([noise])
        # optimizer = torch.optim.Adam([noise], lr=0.1)

        # Define the number of iterations for optimization

        lambda_recon = 100.0
        lambda_adv =  0.05 * 40

        generator.eval()
        discriminator.eval()

        for iteration in range(NUM_ITERATIONS):
            optimizer.zero_grad()
            # Generate an image from the latent vector z
            generated_img = generator(noise)
            discriminator_output = discriminator(generated_img)
            discriminator70_output = discriminator70(generated_img)
            
            # Compute the loss between the generated image and the corrupted image
            recon_loss = criterion(generated_img * mask, masked_img)

            adv_loss = bce_criterion(
                discriminator_output, torch.ones_like(discriminator_output)
            )

            # Total loss
            loss = lambda_recon * recon_loss + lambda_adv * adv_loss
            # Backpropagate the loss
            loss.backward()

            # Update the latent vector z
            optimizer.step()

            # Print the loss every 1000 iterations
            if iteration == NUM_ITERATIONS - 1 or iteration % 50 == 0:
                # Apply the mask
                save_image(
                    generated_img, f"working\\source.jpg", nrow=5, normalize=True
                )
                save_image(
                    original_blacked_img,
                    f"working\\bg.jpg",
                    nrow=5,
                    normalize=True,
                )
                save_image(org_mask_cuda, f"working\\mask.jpg", nrow=5, normalize=True)
                print("d_output", discriminator_output)
                print("d_old_output", discriminator70_output)
                target_path = f"working\\bg.jpg"
                source_path = f"working\\source.jpg"
                mask_path = f"working\\mask.jpg"
                blend_all(target_path, source_path, mask_path, img_num)

                display_img = Image.open(f"working\\output-{img_num}.jpg")
                display(display_img)
                print(
                    f"Iteration [{iteration+1}/{NUM_ITERATIONS}], Loss: {loss.item()}"
                )
