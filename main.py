import os
from PyQt6 import QtGui, uic
from PyQt6.QtWidgets import QApplication, QFileDialog, QMainWindow
from register_two_images import register_two_images
from register_multiple_images import register_multiple_images, make_csv_from_reg_dict
from input_output import display_nth_image_from_tensor
import skimage.io
import torch
from matplotlib import pyplot as plt
import numpy as np
import logging


class Application(object):

    def __init__(self, app):
        self.registration_window = None
        self.results_window = None
        self.ui = None
        self.window = QMainWindow()
        self.app = app
        self.load_main_menu()
        self.dataset_reg_results = None

        self.run()

    def run(self):
        self.app.exec()

    def load_main_menu(self):
        self.ui = uic.loadUi('ui\\mainwindow.ui', self.window)
        self.ui.register_two_images.clicked.connect(self.register_two)
        self.ui.register_dataset.clicked.connect(self.register_dataset)
        self.window.show()

    # load window with results of registration of two images
    # takes registered image as np.array, its width and height as ints as input
    def load_reg_results_window(self, registered_im, width, height):
        logging.debug("entered load_reg-results_window")
        self.results_window = uic.loadUi('ui\\registration_of_two_results.ui', self.window)
        if registered_im is not None:
            if registered_im.ndim == 2:
                QI = QtGui.QImage(registered_im, width, height, QtGui.QImage.Format.Format_Indexed8)
                self.results_window.registered_image.setPixmap(QtGui.QPixmap.fromImage(QI))
            elif registered_im.ndim == 3:
                logging.debug("noticed image is RGB")
                QI = QtGui.QImage(registered_im, width, height, QtGui.QImage.Format.Format_RGB888)
                self.results_window.registered_image.setScaledContents(True)
                self.results_window.registered_image.setPixmap(QtGui.QPixmap.fromImage(QI))
        self.results_window.main_menu.clicked.connect(self.load_main_menu)
        self.results_window.save_result.clicked.connect(lambda: self.save_registered_image(registered_im))
        self.window.show()

    # creates window where user selects two images to be registered
    def register_two(self):
        self.registration_window = uic.loadUi('ui\\register_two_images_mw.ui', self.window)
        self.registration_window.select_reference.clicked.connect(self.browse_reference)
        self.registration_window.select_sample.clicked.connect(self.browse_sample)
        self.registration_window.register_2.clicked.connect(self.perform_registration)
        print("Opening register two dialog.")
        print(self.registration_window)
        # self.registration_window.reference_line_edit.insert("Zde bude cesta k referenci.")
        self.registration_window.show()

    # opens dialog to select reference image for registration
    def browse_reference(self):
        im_filter = 'Images (*.jpg *.jpeg *.png *.tif *.bmp)'
        answer = QFileDialog.getOpenFileName(
            parent=self.registration_window,
            caption='Select reference image',
            directory=os.getcwd(),
            filter=im_filter,
            initialFilter=im_filter
        )
        self.registration_window.reference_line_edit.clear()
        self.registration_window.reference_line_edit.insert(str(answer[0]))

    # opens dialog to select moving image for registration
    def browse_sample(self, write_to):
        im_filter = 'Images (*.jpg *.jpeg *.png *.tif *.bmp)'
        answer = QFileDialog.getOpenFileName(
            parent=self.registration_window,
            caption='Select moving image',
            directory=os.getcwd(),
            filter=im_filter,
            initialFilter=im_filter
        )
        self.registration_window.sample_line_edit.clear()
        self.registration_window.sample_line_edit.insert(str(answer[0]))

    def save_registered_image(self, image):
        path = QFileDialog.getSaveFileName(
            parent=self.window,
            caption='Save registered image',
            directory=os.getcwd()
        )
        print(path[0])
        if len(image.shape) == 2:
            skimage.io.imsave(path[0], image)
        else:
            skimage.io.imsave(path[0], image.transpose((1, 2, 0)))

    # creates window where user selects datasets for registration
    def register_dataset(self):
        self.registration_window = uic.loadUi('ui\\register_datasets.ui', self.window)
        self.registration_window.select_reference.clicked.connect(self.browse_reference_dataset)
        self.registration_window.select_sample.clicked.connect(self.browse_sample_dataset)
        self.registration_window.register_datasets.clicked.connect(self.perform_dataset_registration)
        self.registration_window.main_menu.clicked.connect(self.load_main_menu)
        pass

    # allows user to select folder with reference images
    def browse_reference_dataset(self):
        answer = QFileDialog.getExistingDirectory(
            parent=self.window,
            caption='Select reference folder',
            directory=os.getcwd()
        )
        print(answer)
        self.registration_window.reference_line_edit.clear()
        self.registration_window.reference_line_edit.insert(str(answer))

    # allows user to select folder with moving images
    def browse_sample_dataset(self):
        answer = QFileDialog.getExistingDirectory(
            parent=self.window,
            caption='Select moving images folder',
            directory=os.getcwd()
        )
        print(answer)
        self.registration_window.sample_line_edit.clear()
        self.registration_window.sample_line_edit.insert(str(answer))

    # performs registration of multiple pairs of images
    def perform_dataset_registration(self):
        self.dataset_reg_results = register_multiple_images(self.registration_window.reference_line_edit.text(),
                                                            self.registration_window.sample_line_edit.text())
        print("dataset registration finished")
        self.load_dataset_reg_results_window()

    # displays window of dataset registration results
    # - currently just mention, that registration is completed and possibility to save results
    def load_dataset_reg_results_window(self):
        uic.loadUi('ui\\dataset_save.ui', self.window)
        self.window.file_saved_label.setHidden(True)
        self.window.main_menu.clicked.connect(self.load_main_menu)
        self.window.save_result.clicked.connect(self.save_dataset_reg_result)
        self.window.show()

    # saves results of dataset registration to csv
    # the csv contains translations in x and y and rotations
    def save_dataset_reg_result(self):
        path = QFileDialog.getSaveFileName(
            parent=self.window,
            caption='Save CSV with results',
            directory=os.getcwd()
        )
        print(path[0])
        make_csv_from_reg_dict(self.dataset_reg_results, path[0])
        self.window.file_saved_label.setHidden(False)

    # performs registration of two images and calls function that shows results
    def perform_registration(self):
        # call function that registers images
        results_dict = register_two_images(self.registration_window.reference_line_edit.text(),
                                           self.registration_window.sample_line_edit.text())
        # get registered image and reformat it as a np.array  of uint8
        registered_tens = results_dict["registered_tens"]
        width = registered_tens.shape[2]
        height = registered_tens.shape[3]
        # if image is grayscale
        if registered_tens.shape[1] == 1:
            registered_tens = torch.reshape(registered_tens, (width, height))
            registered_im = np.array(registered_tens)
            registered_im = (registered_im * 255).astype(np.uint8)
            # display registered image
            self.load_reg_results_window(registered_im, width, height)
            self.show_green_purple(self.registration_window.reference_line_edit.text(), registered_im)
            return
        # if image is RGB
        display_nth_image_from_tensor(registered_tens)
        registered_tens = (registered_tens * 255).to(torch.uint8)
        # registered_tens = registered_tens[0, 0, :, :] + \
        #                   registered_tens[0, 1, :, :] * 256 + \
        #                   registered_tens[0, 2, :, :] * 256 ** 2
        registered_tens = registered_tens.permute(0, 2, 3, 1)
        registered_tens = torch.reshape(registered_tens, (3, width, height))
        registered_im = np.array(registered_tens).astype(np.uint8)  # TADY POTREBUJU 24 bit cislo misto tri uint8
        self.load_reg_results_window(registered_im, width, height)

    # show comparison of reference and registered image, where reference image is in green color channel
    # and registered image in red and blue color channels
    def show_green_purple(self, ref_path, result):
        print(ref_path)
        ref = skimage.io.imread(ref_path, as_gray=True)
        green_purple = np.stack((result, ref, result), axis=2)
        QI = QtGui.QImage(green_purple, ref.shape[0], ref.shape[1], QtGui.QImage.Format.Format_RGB888)
        self.results_window.green_purple.setPixmap(QtGui.QPixmap.fromImage(QI))
        pass


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    status = Application(app)
    sys.exit(status)
