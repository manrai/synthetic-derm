import pandas as pd
import torch
from fastai.vision.all import *
from sklearn.model_selection import train_test_split
import wandb
from fastai.callback.wandb import *
from statsmodels.stats.proportion import proportion_confint
from itertools import product
import os

from datasets import load_dataset, Image
from synderm.splits.train_test_splitter import synthetic_train_val_split

# Load in the labels we are using for training data
# TODO: these should be stored in the huggingface repo
metadata_train = pd.read_csv("/n/data1/hms/dbmi/manrai/derm/Fitzpatrick17k/fitzpatrick17k_10label_clean_training.csv")
top_n_labels = metadata_train["label"].value_counts().index[:9]
metadata_train = metadata_train[metadata_train["label"].isin(top_n_labels)].reset_index(drop=True)
metadata_train["synthetic"] = False

# These are the labels we are using for testing
metadata_test = pd.read_csv("/n/data1/hms/dbmi/manrai/derm/Fitzpatrick17k/fitzpatrick17k_10label_clean_held_out_set.csv")
metadata_test = metadata_test[metadata_test["label"].isin(top_n_labels)].reset_index(drop=True)
metadata_test["synthetic"] = False

ids_train = set(metadata_train["md5hash"])
ids_test = set(metadata_test["md5hash"])

if ids_train.isdisjoint(ids_test):
    print("train/test mutually exclusive.")
else:
    print("train/test not mutually exclusive.")


# Train the model
# Set image directory and fastai path
image_dir = "/n/data1/hms/dbmi/manrai/derm/"
path = Path(image_dir)

# Set the random seed
random_state = 111108

# Set the generation folder
generation_folder = "all_generations/finetune-inpaint/"
generation_type = "inpaint"



n_synthetic_per_real = 10

# First, the dataset is duplicated n_synthetic_per_real times
df = pd.concat([metadata_train]*n_synthetic_per_real, ignore_index=True)

# create a variable that represents the nth copy of the image
df['n'] = df.groupby('md5hash').cumcount()
df['location'] = generation_folder + df['label'].str.replace(' ', '-')  + '/' + generation_type +'/0' + df['n'].astype(str) + '/' + df['md5hash'] + '.png'
df['synthetic'] = True
df['Qc'] = ''

# drop the 'n' column
df = df.drop(columns=['n'])

train, val = synthetic_train_val_split(
    real_data=metadata_train, 
    synthetic_data = df, 
    per_class_test_size=40,
    n_real_per_class = 32,
    random_state = random_state,
    class_column = "label",
    mapping_real_to_synthetic = "md5hash")


print(train.shape[0])
print(val.shape[0])