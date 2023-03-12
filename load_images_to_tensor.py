import os
import numpy as np
import skimage.io
import torch
import matplotlib.pyplot as plt


def load_grayscale_images_to_tensor(path, datatype=torch.float32):
    # function takes a path to folder with images
    # requires only grayscale images of the same size to be present in that folder
    # function loads all images in folder to tensor of size number_of_images x 1 x height x width of the image
    all_files = os.listdir(path)
    im = skimage.io.imread(os.path.join(path, all_files[0]))
    height = im.shape[-2]
    width = im.shape[-1]
    batch_size = len(all_files)
    output_tensor = torch.empty((batch_size, 1, height, width), dtype=datatype)

    for i, f in enumerate(all_files):
        im = skimage.io.imread(os.path.join(path, f))
        output_tensor[i, 0, :, :] = torch.tensor(im, dtype=datatype) / 256
    return output_tensor


def load_rgb_images_to_tensor(path, datatype=torch.float32):
    # function takes a path to folder with images
    # requires only rgb images of the same size to be present in that folder
    # function loads all images in folder to tensor of size number_of_images x 1 x height x width of the image
    all_files = os.listdir(path)
    im = skimage.io.imread(os.path.join(path, all_files[0]))
    print(im.shape)
    height = im.shape[0]
    width = im.shape[1]
    batch_size = len(all_files)
    output_tensor = torch.empty((batch_size, 3, height, width), dtype=datatype)
    for i, f in enumerate(all_files):
        im = skimage.io.imread(os.path.join(path, f))
        im_tensor = torch.tensor(im, dtype=datatype) / 255
        output_tensor[i, :, :, :] = im_tensor.view(im_tensor.shape[2], im_tensor.shape[0], im_tensor.shape[1])
    return output_tensor


def display_fst_image_from_tensor(tensor):
    # this function displays first image from tensor
    tensor_image = tensor[0].view(tensor.shape[2], tensor.shape[3], tensor.shape[1])
    plt.imshow(tensor_image)
    plt.show()
