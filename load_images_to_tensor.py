import os
import numpy as np
import skimage.io
import torch
import matplotlib


def load_images_to_tensor(path, infer_dimesions=True, width=1024, height=1024):
    all_files = os.listdir(path)
    im = skimage.io.imread(os.path.join(path, all_files[0]))
    # skimage.io.imshow(im)
    if infer_dimesions:
        im = skimage.io.imread(os.path.join(path, all_files[0]))
        height = im.shape[-2]
        width = im.shape[-1]
    batch_size = len(all_files)
    output_tensor = torch.empty((batch_size, 1, height, width))

    for i, f in enumerate(all_files):
        im = skimage.io.imread(os.path.join(path, f))
        output_tensor[i, 0, :, :] = torch.tensor(im)
    return output_tensor
