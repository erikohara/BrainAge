import numpy as np
import pandas as pd
import os
import torch
import sys

from SFCN import SFCNModel
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
from monai.data import ImageDataset, DataLoader
from monai.transforms import EnsureChannelFirst, Compose, NormalizeIntensity

DEBUG = False


def print_title(str):
    print("\n", "-" * 10, f" {str} ", "-" * 10)


def list_avg(ls):
    return sum(ls) / len(ls)


def read_data(path, postfix, max_entries=-1, normalize=False):
    '''
        Reads the images from a directory and returns the valid image paths, normalized ages, and
        a denormalize function

        Parameters
        ----------
        folder_name : string
                Folder containing all of the images. The folder must be in the same directory as the script.
        
        postfix : string
                Constant string in the filenames that is followed by the user id.
        
        max_entries : int (optional)
                Maximum number of images to be read. Defaults to reading the entire directory.
        
        Returns
        -------
        images : python list
            A list containing paths of all the images.

        mean_age : float
            Average age of the subjects sampled
        
        norm_ages : python list
            Normalized version of the ages.
        
        denorm_fn : python function
            A function that can be used to convert the normalized
            values to the ages.
        
    '''

    df = pd.read_csv('/home/erik.ohara/BrainAge/ukbb_img.csv')
    # df = pd.read_csv('/home/finn.vamosi/3Brain/ukbb_img.csv')

    images = []
    ages = np.array([])
    diseased = []
    idx = 0
    ids = []

    # Cleaning the data
    df = df.drop_duplicates(subset=["EID"])

    with open("/home/erik.ohara/BrainAge/overlap.txt", "r") as file:
        for line in file:
            # remove linebreak (which is last character)
            name = line[:-1]
            diseased.append(name)

        for f in sorted(os.listdir(path)):
            if f.endswith(".nii.gz"):
                # Find the EID in the file
                filename = f.split('_')[0]

                if filename == diseased[idx]:
                    # skip over diseased subjects
                    if idx < len(diseased) - 1:
                        idx += 1
                else:
                    images.append(f)
                    ids.append(filename)

                    # Find the corresponding age
                    age = f.split('_')[1]
                    age = np.float32(age)
                    ages = np.append(ages, age)

    # Convert the images into paths
    images = [os.sep.join([path, image]) for image in images]

    # Z Normalizing the ages
    mean_age = ages.mean()
    sd_age = ages.std()
    norm_ages = (ages - mean_age) / sd_age

    print(len(images))

    # Creating a function that can be used for converting
    # the normalized number back to the original ages
    # denorm_fn = lambda x : x*sd_age + mean_age
    if not normalize:
        denorm_fn = lambda x: x
        norm_ages = ages

    with open("cfs.txt", "w") as file:
        print("Saving counterfactual ids to 'cfs.txt'...")

        for name in ids:
            file.write("%s\n" % name)
        print("Done.")

    if DEBUG:
        print_title("Reading the CSV File")
        print(df)
        # print(images[:5])
        print(ages[:5])
        print("mean age: ", mean_age, " sd age: ", sd_age)
        print("Displaying the frequency distribution of the data...")
        plt.hist(ages, bins=50)
        plt.gca().set(title='Frequency Histogram', ylabel='Frequency', xlabel='Ages')
        plt.show()

    return images, mean_age, norm_ages, denorm_fn


def MAE_with_mean_fn(mean, ls):
    AE_list = [abs(x - mean) for x in ls]
    return list_avg(AE_list)
