import pandas as pd
import torch
from fastai.vision.all import *
from sklearn.model_selection import train_test_split
import wandb
from fastai.callback.wandb import *
from statsmodels.stats.proportion import proportion_confint
from itertools import product
import os

# Set image directory and fastai path
image_dir = "/n/data1/hms/dbmi/manrai/derm/"
path = Path(image_dir)

wandb_project = "dose_response_june7"

# Set the random seed
random_state = 119108

# Set the generation folder
generation_folder = "generations-more/text-to-image/"

# Set parameters for the experiment
n_per_label_list = [128, 228]
include_synthetic_list = [True]
n_synthetic_per_real_list = [10, 25, 50, 75]
generation_type_list = ['text-to-image']

# Create a dictionary to store the different types of augmentation
augmentation_dict = {
    "None": [],
    "aug_transforms": aug_transforms()
}


# a function that takes in a dataframe and returns a dataframe that consists of just n_synthetic_per_real copies of each line in the dataframe
# The function also adds a column called location which is the location of the image and a column called synthetic which is True
def generate_synthetic_metadata(train_val_df, generation_type, n_synthetic_per_real=10, folder=generation_folder):
    df = pd.concat([train_val_df]*n_synthetic_per_real, ignore_index=True)
    # create a variable that represents the nth copy of the image
    df['n'] = df.groupby('md5hash').cumcount() + 10
    df['location'] = folder + df['label'].str.replace(' ', '-')  + '/' + generation_type + '/' + df['n'].astype(str) + '/' + df['md5hash'] + '.png'
    df['synthetic'] = True
    df['Qc'] = ''
    # drop the 'n' column
    df = df.drop(columns=['n'])
    return df    

# A function that takes in the train data and 2 arguments: n_per_label and include_synthetic.
# It selects n_per_label real images from each label and returns a dataframe with the selected images, concatenated with the val data
# If include_synthetic is True, it also includes the synthetic images that share the same md5hash as the real images selected
def get_n_per_label_data(train_data, val_data, n_per_label, random_state, include_synthetic=False):
    # Select n_per_label real images from each label
    real_data = train_data[train_data['synthetic'] == False]
    n_per_label_data = pd.DataFrame(real_data.groupby(['label']).apply(lambda x: x.sample(n=n_per_label, random_state=random_state)).reset_index(drop=True))
    if include_synthetic:
        # Get the md5hashes of the real images selected
        md5hashes = n_per_label_data['md5hash']
        # Get the synthetic images that share the same md5hash as the real images selected
        n_per_label_data = train_data[train_data['md5hash'].isin(md5hashes)]
    # Concatenate the n_per_label data with the val data
    n_per_label_data = pd.concat([n_per_label_data, val_data])
    return n_per_label_data


if __name__ == "__main__":

    # Read in the metadata
    metadata = pd.read_csv('../Metadata/fitzpatrick17k_10label_clean_training.csv')

    # Filter to the top 10 most common labels
    top_n_labels = metadata['label'].value_counts().index[:9]
    metadata = metadata[metadata['label'].isin(top_n_labels)].reset_index(drop=True)
    metadata['location'] = 'Fitzpatrick17k/finalfitz17k/' + metadata['md5hash'] + '.jpg'
    metadata['synthetic'] = False

    # Read in the test data
    test_data = pd.read_csv("../Metadata/fitzpatrick17k_10label_clean_held_out_set.csv")
    test_data = test_data[test_data['label'].isin(top_n_labels)].reset_index(drop=True)
    test_data['location'] = 'Fitzpatrick17k/finalfitz17k/' + test_data['md5hash'] + '.jpg'
    test_data['synthetic'] = False
    test_data['is_valid'] = False


    # Loop through the possible values for function parameters
    for combo in itertools.product(n_per_label_list, include_synthetic_list, n_synthetic_per_real_list, generation_type_list, augmentation_dict):
        n_per_label, include_synthetic, n_synthetic_per_real, generation_type, augmentation = combo

        # Make sure no train data is in the test set
        train_val_data = pd.DataFrame(metadata[~metadata['md5hash'].isin(test_data['md5hash'])])
        # Get the synthetic metadata
        synthetic_metadata = generate_synthetic_metadata(train_val_data, generation_type=generation_type, n_synthetic_per_real=n_synthetic_per_real, folder = generation_folder)

        # Concatenate the synthetic data with the train/val data
        train_val_data = pd.concat([train_val_data, synthetic_metadata]).reset_index(drop=True)

        # Split the train/val data into train and val, with 40 images from each label in the val set
        real_train_val_data = train_val_data[train_val_data['synthetic'] == False]
        val_data = pd.DataFrame(real_train_val_data.groupby(['label']).apply(lambda x: x.sample(n=40, random_state=random_state, replace=False)).reset_index(drop=True))
        train_data = pd.DataFrame(train_val_data[~train_val_data['md5hash'].isin(val_data['md5hash'])])

        # Add an 'is_valid' column to the train and val data
        train_data['is_valid'] = False
        val_data['is_valid'] = True

        # Get the data
        df = get_n_per_label_data(train_data, val_data, n_per_label, random_state, include_synthetic)

        # Print a summary of the run data
        print("Number of real training images per label:", len(df[(df['synthetic']==False) & (df['is_valid']==False)]) /df['label'].nunique())
        print("Number of synthetic training images per label:",  len(df[(df['synthetic']==True) & (df['is_valid']==False)]) /df['label'].nunique())

        # adjust batch size based on number of images
        if (len(df[df.is_valid == False])/10 >= 500):
            batch_size = 128
        elif (len(df[df.is_valid == False])/10 >= 100):
            batch_size = 64
        elif (len(df[df.is_valid == False])/10 >= 10):
            batch_size = 32
        else:
            batch_size = 8

        # Create a fastai dataloader
        dls = ImageDataLoaders.from_df(df, 
                                path,
                                fn_col='location',
                                label_col='label',
                                valid_col='is_valid', 
                                bs=batch_size,
                                item_tfms=Resize(224),
                                batch_tfms=augmentation_dict[augmentation])            
        # Set config parameters for wandb
        config = dict (
            architecture = "EfficientNet-V2-M",
            gen_folder = generation_folder,
            random_state = random_state,
            augmentation = augmentation, 
            n_training_per_label = n_per_label,
            include_synthetic = include_synthetic,
            n_synthetic_per_real = n_synthetic_per_real,
            generation_type = generation_type
        )
        # set tags for wandb and
        sample_tag = "n_real_per_label_" + str(n_per_label)
        seed_tag = "seed_" + str(random_state)
        include_synthetic_tag = "include_synthetic_" + str(include_synthetic)
        generation_type_tag = str(generation_type)

        wandb.init(
        project=wandb_project,
        tags=[sample_tag, seed_tag, include_synthetic_tag, generation_type_tag],
        config=config,
        )

        learn = vision_learner(dls, 
                            arch=efficientnet_v2_m,                           
                            metrics=[error_rate, accuracy])
        # fit with wandb callback
        learn.fit(50, cbs=[WandbCallback(), EarlyStoppingCallback (monitor='valid_loss', min_delta=0.0, patience=3),
                           SaveModelCallback (monitor='valid_loss', fname='best_model')])
        # load the best model
        learn.load('best_model')

        # get the confusion matrix and log it to wandb
        # interp = ClassificationInterpretation.from_learner(learn)
        # cm_plot = interp.plot_confusion_matrix(title=f"N per label:{n_per_label} | Synthetic:{include_synthetic} | Val set")
        # wandb.log({"Validation Confusion Matrix": interp.confusion_matrix()})

        # predict on test data
        test_dl = dls.test_dl(test_data)
        # get predictions and probabilities for test set
        preds, _ = learn.get_preds(dl=test_dl)
        # get predicted labels for top-1 and top-3
        top1_pred = torch.argmax(preds, dim=1)
        top3_pred = torch.topk(preds, k=3, dim=1).indices

        md5hashes = test_data['md5hash']
        # get predicted labels and probabilities for top-1 and top-3
        top1_prob, top1_label = torch.topk(preds, k=1, dim=1)
        top3_prob, top3_label = torch.topk(preds, k=3, dim=1)

        # convert tensor labels to class labels
        top1_label = [learn.dls.vocab[i] for i in top1_label.squeeze()]
        top3_label = [[learn.dls.vocab[j] for j in i] for i in top3_label]

        # get true labels for test set
        true_labels = test_data['label']

        # calculate accuracy scores
        top1_acc = (top1_label == true_labels).mean()
        top3_acc = torch.zeros(len(true_labels))
        for i in range(len(true_labels)):
            top3_acc[i] = true_labels[i] in top3_label[i]
        top3_acc = top3_acc.mean()

        # calculate upper and lower bounds for 95% confidence interval
        top1_ci_lower, top1_ci_upper = proportion_confint(top1_acc*len(true_labels), len(true_labels), alpha=0.05, method='normal')
        top3_ci_lower, top3_ci_upper = proportion_confint(top3_acc*len(true_labels), len(true_labels), alpha=0.05, method='normal')

        # log accuracy scores to wandb
        wandb.log({'top1_acc': top1_acc,
                    'top1_ci_lower': top1_ci_lower,
                    'top1_ci_upper': top1_ci_upper,
                    'top3_acc': top3_acc,
                    'top3_ci_lower': top3_ci_lower,
                    'top3_ci_upper': top3_ci_upper})

        # split up the top3 probabilities
        top1_prob, top2_prob, top3_prob = torch.split(top3_prob, 1, dim=1)

        # Convert the tensors to NumPy arrays
        top1_prob_arr = top1_prob.numpy().flatten()
        top2_prob_arr = top2_prob.numpy().flatten()
        top3_prob_arr = top3_prob.numpy().flatten()

        # split up the top3 labels to match 
        top1_label = [sublist[0] for sublist in top3_label]
        top2_label = [sublist[1] for sublist in top3_label]
        top3_label = [sublist[2] for sublist in top3_label]

        # create dataframe of predictions
        df_pred = pd.DataFrame({
            'architecture' : "EfficientNet-V2-M",
            'random_state' : random_state,
            'augmentation' : augmentation,
            'gen_folder' : generation_folder,
            'generation_type' : generation_type,
            'n_training_per_label' : n_per_label,
            'n_synthetic_per_real' : n_synthetic_per_real,
            'include_synthetic' : include_synthetic,
            'md5hash': md5hashes,
            'true_label': true_labels,
            'top1_label': top1_label,
            'top1_prob': top1_prob_arr,
            'top2_label': top2_label,
            'top2_prob': top2_prob_arr,
            'top3_label': top3_label
        })

        # log the test predictions
        wandb.log({"test_predictions": wandb.Table(dataframe=df_pred)})

        # Finish the run
        wandb.finish()