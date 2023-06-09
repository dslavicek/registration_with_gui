import torch
import load_images_to_tensor as lit
import torch.nn.functional as F

# class for generating synthetic samples for registration
class SampleCreator:
    def __init__(self, input_images, verbose=False, from_path=True):
        self.verbose = verbose
        self.datatype = torch.float32
        self.x_shifts = [-0.1, 0, 1]
        self.y_shifts = [-0.1, 0, 1]
        self.rotations_deg = [-1, 0, 1]
        if from_path:
            self.input_tensor = lit.load_grayscale_from_folder(input_images)
        else:
            self.input_tensor = input_images
        self.batch_size = self.input_tensor.shape[0]
        self.height = self.input_tensor.shape[2]
        self.width = self.input_tensor.shape[3]
        self.output = None
        self.transf_matrices = None
        if verbose:
            print("Object of class SampleCreator created. Batch size: " + str(self.batch_size) + ", height: " +
                  str(self.height) + ", width:" + str(self.width))

    def check_params_length(self):
        if (len(self.x_shifts) == self.batch_size and len(self.y_shifts) == self.batch_size and
                len(self.rotations_deg) == self.batch_size):
            if self.verbose:
                print("Lengths of transformation parameter arrays are of suitable length.")
            return 0
        print("Error: transformation parameter arrays have wrong lengths.")
        return 1

    def generate_samples(self):
        if self.check_params_length():
            return 1
        output = self.input_tensor.clone()
        t_mats = torch.empty(size=(self.batch_size, 2, 3), dtype=self.datatype)
        x_shift_tens = torch.tensor(self.x_shifts, dtype=self.datatype)
        y_shift_tens = torch.tensor(self.y_shifts, dtype=self.datatype)
        rot_tens_rad = torch.tensor(self.rotations_deg, dtype=self.datatype) / 180 * torch.pi
        rot_sin = torch.sin(rot_tens_rad)
        rot_cos = torch.cos(rot_tens_rad)
        t_mats[:, 0, 0] = rot_cos
        t_mats[:, 0, 1] = -rot_sin
        t_mats[:, 1, 0] = rot_sin
        t_mats[:, 1, 1] = rot_cos
        t_mats[:, 0, 2] = x_shift_tens
        t_mats[:, 1, 2] = y_shift_tens

        grid = F.affine_grid(t_mats, [self.batch_size, 1, self.height, self.width])
        output = F.grid_sample(output, grid, padding_mode="reflection")
        self.output = output
        self.transf_matrices = t_mats.detach()
        return output
