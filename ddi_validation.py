import pandas as pd
import torch
from fastai.vision.all import *
from sklearn.model_selection import train_test_split
import wandb
from fastai.callback.wandb import *
from statsmodels.stats.proportion import proportion_confint
from itertools import product
import os

# Utilize the second GPU if available
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# torch.cuda.empty_cache()

# Set image directory and fastai path
image_dir = "/n/data1/hms/dbmi/manrai/derm/"
path = Path(image_dir)

wandb_project = "n_real_per_disease_DDI_xxx"
# Set the random seed
random_state = 94894

# Set the generation folder
generation_folder = "generations-inpaint-ddi/all"

# Set parameters for the experiment
n_per_label_list = [-1]
include_synthetic_list = [False, True]
n_synthetic_per_real_list = [10]
generation_type_list = ['inpaint', 'inpaint-outpaint']

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
    df['n'] = df.groupby('DDI_ID').cumcount()
    df['location'] = folder + '/' + generation_type +'/0' + df['n'].astype(str) + '/' + df['DDI_file']
    df['synthetic'] = True
    df['Qc'] = ''
    # drop the 'n' column
    df = df.drop(columns=['n'])
    return df    

# A function that takes in the train data and 2 arguments: n_per_label and include_synthetic.
# It selects n_per_label real images from each label and returns a dataframe with the selected images, concatenated with the val data
# If include_synthetic is True, it also includes the synthetic images that share the same ID as the real images selected
def get_n_per_label_data(train_data, val_data, random_state, n_per_label=-1, include_synthetic=False):
    # Select n_per_label real images from each label
    real_data = train_data[train_data['synthetic'] == False]
    if n_per_label == -1:
        n_per_label_data = real_data
    else:
        n_per_label_data = pd.DataFrame(real_data.groupby(['disease']).apply(lambda x: x.sample(n=n_per_label, random_state=random_state)).reset_index(drop=True))
    if include_synthetic:
        # Get the IDs of the real images selected
        ddi_ids = n_per_label_data['DDI_ID']
        # Get the synthetic images that share the same ID as the real images selected
        n_per_label_data = train_data[train_data['DDI_ID'].isin(ddi_ids)]
    # Concatenate the n_per_label data with the val data
    n_per_label_data = pd.concat([n_per_label_data, val_data])
    return n_per_label_data

if __name__ == "__main__":
    # Read in the metadata
    metadata = pd.read_csv('../Metadata/ddi_training.csv')
    # metadata = metadata[metadata['disease'].isin(disease_names_ddi)].reset_index(drop=True)
    metadata['location'] = 'Stanford_DDI/ddi_images/' + metadata['DDI_file']
    metadata['synthetic'] = False
    metadata['FST_bin'] = metadata['skin_tone'].apply(lambda x: "FST_12" if x in [12] else "FST_34" if x in [34] else "FST_56" if x in [56] else 'unknown')


    # Take out 40 images from each label for testing
    # test_data = pd.DataFrame(metadata.groupby(['FST_bin']).apply(lambda x: x.sample(n=32, random_state=random_state)).reset_index(drop=True)) 
    test_data = pd.read_csv("../Metadata/ddi_heldout.csv")
    test_data['is_valid'] = False

    # Remove the test data from the train/val data
    train_val_data = pd.DataFrame(metadata[~metadata['DDI_ID'].isin(test_data['DDI_ID'])])

    synthetic_metadata = generate_synthetic_metadata(train_val_data, generation_type='inpaint', n_synthetic_per_real=10, folder = generation_folder)

    # Concatenate the synthetic data with the train/val data
    train_val_data = pd.concat([train_val_data, synthetic_metadata]).reset_index(drop=True)

    # Split the train/val data into train and val, with 32 images from each label in the val set
    real_train_val_data = train_val_data[train_val_data['synthetic'] == False]
    val_data = pd.DataFrame(real_train_val_data.groupby(['FST_bin']).apply(lambda x: x.sample(n=32, random_state=random_state, replace=False)).reset_index(drop=True))
    train_data = pd.DataFrame(train_val_data[~train_val_data['DDI_ID'].isin(val_data['DDI_ID'])])

    # Add an 'is_valid' column to the train and val data
    train_data['is_valid'] = False
    val_data['is_valid'] = True
    real_train_val_data.shape
    real_train_val_data.groupby(['malignant', 'FST_bin']).size()
    metadata[metadata['disease'].isin(disease_names_ddi)].groupby(['malignant', 'disease']).size()
    metadata.groupby(['malignant', 'FST_bin']).size()
   
    # show the number of real and synthetic images in the train data, val data, and test data
    print("Number of real images in train data: ", len(train_data[train_data['synthetic'] == False]))
    print("Number of synthetic images in train data: ", len(train_data[train_data['synthetic'] == True]))
    print("Number of real images in val data: ", len(val_data[val_data['synthetic'] == False]))
    print("Number of synthetic images in val data: ", len(val_data[val_data['synthetic'] == True]))
    print("Number of real images in test data: ", len(test_data[test_data['synthetic'] == False]))
    print("Number of synthetic images in test data: ", len(test_data[test_data['synthetic'] == True]))

    # Loop through the possible values for function parameters
    for combo in itertools.product(n_per_label_list, include_synthetic_list, n_synthetic_per_real_list, generation_type_list, augmentation_dict):
        n_per_label, include_synthetic, n_synthetic_per_real, generation_type, augmentation = combo
        print("running for:", combo)
        # Make sure no train data is in the test set
        train_val_data = pd.DataFrame(metadata[~metadata['DDI_ID'].isin(test_data['DDI_ID'])])
        # Get the synthetic metadata
        synthetic_metadata = generate_synthetic_metadata(train_val_data, generation_type=generation_type, n_synthetic_per_real=n_synthetic_per_real, folder = generation_folder)

        # Concatenate the synthetic data with the train/val data
        train_val_data = pd.concat([train_val_data, synthetic_metadata]).reset_index(drop=True)

        # Split the train/val data into train and val, with 40 images from each label in the val set
        real_train_val_data = train_val_data[train_val_data['synthetic'] == False]
        val_data = pd.DataFrame(real_train_val_data.groupby(['FST_bin']).apply(lambda x: x.sample(n=32, random_state=random_state, replace=False)).reset_index(drop=True))
        train_data = pd.DataFrame(train_val_data[~train_val_data['DDI_ID'].isin(val_data['DDI_ID'])])

        # Add an 'is_valid' column to the train and val data
        train_data['is_valid'] = False
        val_data['is_valid'] = True
        print(n_per_label)
        # Get the data
        df = get_n_per_label_data(train_data, val_data, random_state, n_per_label=-1, include_synthetic=include_synthetic)
        # adjust batch size based on number of images
        if (len(df[df.is_valid == False])/10 >= 100):
            batch_size = 64
        elif (len(df[df.is_valid == False])/10 >= 10):
            batch_size = 32
        else:
            batch_size = 8

        # Create a fastai dataloader
        dls = ImageDataLoaders.from_df(df, 
                                path,
                                fn_col='location',
                                label_col='malignant',
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
        learn.fit(50, cbs=[WandbCallback(), EarlyStoppingCallback (monitor='valid_loss', min_delta=0.0, patience=3)])
        # SaveModelCallback (monitor='valid_loss', fname='best_model')
        # load the best model
        # learn.load('best_model')
        
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
        
        ddi_id = test_data['DDI_ID']
        # get predicted labels and probabilities for top-1 
        top1_prob, top1_label = torch.topk(preds, k=1, dim=1)

        # convert tensor labels to class labels
        top1_label = [learn.dls.vocab[i] for i in top1_label.squeeze()]
        
        # get true labels for test set
        true_labels = test_data['malignant']
        
        # calculate accuracy scores
        top1_acc = (top1_label == true_labels).mean()

        # calculate upper and lower bounds for 95% confidence interval
        top1_ci_lower, top1_ci_upper = proportion_confint(top1_acc*len(true_labels), len(true_labels), alpha=0.05, method='normal')
        
        # log accuracy scores to wandb
        wandb.log({'top1_acc': top1_acc,
                    'top1_ci_lower': top1_ci_lower,
                    'top1_ci_upper': top1_ci_upper,
                    })

        # Convert the tensors to NumPy arrays
        top1_prob_arr = top1_prob.numpy().flatten()
        
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
            'ddi_id': ddi_id,
            'true_label': true_labels,
            'top1_label': top1_label,
            'top1_prob': top1_prob_arr
        })
        
        # log the test predictions
        wandb.log({"test_predictions": wandb.Table(dataframe=df_pred)})
        
        # Finish the run
        wandb.finish()
    wandb.finish()

