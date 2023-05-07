from input_output import from_image_to_tensor
from registration import rigid_registration


def register_two_images(path_to_reference, path_to_sample):
    print("Starting registration.")
    ref = from_image_to_tensor(path_to_reference)
    sample = from_image_to_tensor(path_to_sample)
    print("Images loaded")
    return rigid_registration(ref, sample, verbose=True)
