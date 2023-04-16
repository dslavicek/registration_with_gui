from load_images_to_tensor import load_grayscale_from_folder
from registration import grayscale_registration
import pandas as pd
import numpy as np
import os


def register_multiple_images(path_to_reference, path_to_sample):
    print("Starting registration.")
    ref = load_grayscale_from_folder(path_to_reference)
    sample = load_grayscale_from_folder(path_to_sample)
    print("Images loaded")
    return grayscale_registration(ref, sample, verbose=True)


def make_csv_from_reg_dict(registration_dict, output_path):
    x = registration_dict["x_shifts"]
    y = registration_dict["y_shifts"]
    angle = registration_dict["angles_rad"]
    data = np.stack((x, y, angle), axis=1)
    data = np.transpose(data)
    result = pd.DataFrame(data)
    print("X")
    # result.to_csv(os.path.join(output_path, "result.csv"))
    # result.to_csv(os.path.join(output_path, "result2.csv"), index=False)
    result.to_csv(os.path.join(output_path, "result.csv"), header=["x shift", "y shift", "angle"], index=False)
