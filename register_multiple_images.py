from input_output import from_folder_to_tensor
from registration import rigid_registration
import pandas as pd
import numpy as np


def register_multiple_images(path_to_reference, path_to_sample):
    print("Starting registration.")
    ref = from_folder_to_tensor(path_to_reference)
    sample = from_folder_to_tensor(path_to_sample)
    print("Images loaded")
    return rigid_registration(ref, sample, verbose=True)


def make_csv_from_reg_dict(registration_dict, output_path):
    x = registration_dict["x_shifts"]
    y = registration_dict["y_shifts"]
    angle = registration_dict["angles_rad"]
    data = np.stack((x, y, angle), axis=1)
    data = np.transpose(data)
    result = pd.DataFrame(data)
    result.to_csv(output_path, header=["x shift", "y shift", "angle"], index=False)
