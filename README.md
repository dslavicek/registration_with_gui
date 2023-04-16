# registration_with_gui
This repository contains code for my diploma thesis, which is still being developed.
The topic of my diploma thesis is "Image registration using optimization with automatic differentiation".

My diploma thesis has a simple GUI. For now, the application is only in Czech language and supports only grayscale images. When user starts the application they have two options. They can either reister two images or register images from folder. This version supports only registration with rigid transform.

## Registering two images.
The user selects two images - one reference image and one moving image. The images must have same dimensions. The application performs registration and shows registered image and comparison of registered and reference image. Comparison is displayed by putting reference image to green color channel and registered image to red and blue color channels. User can also save the registered image.

## Registering folders
User selects two folders one for reference images, the other for moving images. The folders must contain only images for registration and all images must be of same size. The application than performs registration and user can save CSV file with results. The CSV file contains translations in X and Y directions and rotations. Translations are on scale from -1 to 1, rotation is expressed in radians.

