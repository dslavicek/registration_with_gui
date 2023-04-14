import os
from PyQt6 import QtGui, uic
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QTextEdit, QFileDialog, QMainWindow
from register_two_images import register_two_images

import skimage.io
import torch
from matplotlib import pyplot as plt
import numpy as np


class Application(object):
    # def __init__(self, app):
    #     self.registration_window = None
    #     self.results_window = None
    #     self.window = QMainWindow()
    #     self.ui = uic.loadUi('ui\\mainwindow.ui', self.window)
    #     self.app = app
    #     self.ui.register_two_images.clicked.connect(self.register_two)
    #     self.ui.register_dataset.clicked.connect(self.register_dataset)
    #     print("initializing")
    #     print(self.ui)
    #     # self.ui.show()
    #     self.window.show()
    #     self.run()

    def __init__(self, app):
        self.registration_window = None
        self.results_window = None
        self.ui = None
        self.window = QMainWindow()
        self.app = app
        self.load_main_menu()

        # self.ui.show()
        # self.window.show()
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
        self.results_window = uic.loadUi('ui\\registration_of_two_results.ui', self.window)
        if registered_im is not None:
            QI = QtGui.QImage(registered_im, width, height, QtGui.QImage.Format.Format_Indexed8)
            self.results_window.registered_image.setPixmap(QtGui.QPixmap.fromImage(QI))
        self.results_window.main_menu.clicked.connect(self.load_main_menu)
        self.results_window.save_result.clicked.connect(lambda: self.save_registered_image(registered_im))
        self.window.show()
        pass

    def register_two(self):
        self.registration_window = uic.loadUi('ui\\register_two_images_mw.ui', self.window)
        self.registration_window.select_reference.clicked.connect(self.browse_reference)
        self.registration_window.select_sample.clicked.connect(self.browse_sample)
        self.registration_window.register_2.clicked.connect(self.perform_registration)
        print("Opening register two dialog.")
        print(self.registration_window)
        # self.registration_window.reference_line_edit.insert("Zde bude cesta k referenci.")
        self.registration_window.show()

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
        skimage.io.imsave(path[0], image)

    def register_dataset(self):
        pass

    def perform_registration(self):
        # call function that registers images
        results_dict = register_two_images(self.registration_window.reference_line_edit.text(),
                                           self.registration_window.sample_line_edit.text())
        # get registered image and reformat it as a np.array  of uint8
        registered_tens = results_dict["registered_tens"]
        width = registered_tens.shape[2]
        height = registered_tens.shape[3]
        registered_tens = torch.reshape(registered_tens, (width, height))
        registered_im = np.array(registered_tens)
        registered_im = (registered_im * 255).astype(np.uint8)
        # display registered image
        self.load_reg_results_window(registered_im, width, height)
        self.show_green_purple(self.registration_window.reference_line_edit.text(), registered_im)

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
