#!/usr/bin/env python
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

# Functions and classes for loading and using the Inception model.
import inception


print(tf.__version__)


inception.maybe_download()


model = inception.Inception()


def display(image_path):
    # Display the image
    image = plt.imread(image_path)
    plt.imshow(image)
    plt.show()


def classify(image_path):
    # Display the image.
    display(image_path)

    # Use the Inception model to classify the image.
    pred = model.classify(image_path=image_path)

    # Print the scores and names for the top-10 predictions.
    model.print_scores(pred=pred, k=10, only_first_name=True)


image_path = os.path.join(inception.data_dir, 'cropped_panda.jpg')
classify(image_path)
classify(image_path="images/parrot.jpg")


def plot_resized_image(image_path):
    # Get the resized image from the Inception model.
    resized_image = model.get_resized_image(image_path=image_path)

    # Plot the image.
    plt.imshow(resized_image, interpolation='nearest')

    # Ensure that the plot is shown.
    plt.show()


plot_resized_image(image_path="images/parrot.jpg")
classify(image_path="images/parrot_cropped1.jpg")
classify(image_path="images/parrot_cropped2.jpg")
classify(image_path="images/parrot_cropped3.jpg")
classify(image_path="images/parrot_padded.jpg")
classify(image_path="images/elon_musk.jpg")
classify(image_path="images/elon_musk_100x100.jpg")
plot_resized_image(image_path="images/elon_musk_100x100.jpg")
classify(image_path="images/willy_wonka_old.jpg")
classify(image_path="images/willy_wonka_new.jpg")

model.close()