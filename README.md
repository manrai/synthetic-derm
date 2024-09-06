## Augmenting medical image classifiers with synthetic data from latent diffusion models

Luke W. Sagers*, James A. Diao*, Luke Melas-Kyriazi*, Matthew Groh, Vijaytha Muralidharan, Zhuo Ran Cai, Jesutofunmi A. Omiye, Pranav Rajpurkar, Adewole S. Adamson, Veronica Rotemberg, Roxana Daneshjou, and Arjun K. Manrai

**Abstract:** While hundreds of artificial intelligence (AI) algorithms are now approved or cleared by the US Food and Drugs Administration (FDA), many studies have shown inconsistent generalization or bias, particularly for underrepresented populations. Some have proposed that generative AI could reduce the need for real data, but its utility in model development remains unclear. Skin disease serves as a useful case study in synthetic image generation due to the diversity of disease appearances, particularly across the protected attribute of skin tone. Here we show that latent diffusion models can scalably generate images of skin disease and that augmenting model training with these data improves performance in data-limited settings. These performance gains saturate at synthetic-to-real image ratios above 10:1 and are substantially smaller than the gains obtained from adding real images. We further conducted a human reader study on the synthetic generations, revealing a correlation between physician-assessed photorealism and improvements in model performance. We release a new dataset of 458,920 synthetic images produced using several generation strategies. Our results suggest that synthetic data could serve as a force-multiplier for model development, but the collection of diverse real-world data remains the most important step to improve medical AI algorithms.


### Data
Two datasets were used to generate synthetic images:

The Fitzpatrick17k dataset, curated by Groh et al. 
You can learn more by reading their paper here: https://arxiv.org/abs/2104.09957

The Stanford Diverse Dermatology Images Dataset (DDI), curated by Daneshjou et al. 
You can learn more by reading their paper here: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9374341/


### SynDerm

#### Overview and Installation

SynDerm is our library for synthetic data augmentation of skin disease images using large-scale generative models. The library can also be used to reproduce the primary results of our paper. 

It can be installed with:
```bash
pip install synderm
```

The library is organized as follows:

- `generation/`: Code for synthetic data generation
- `experiments/`: Code for image classification experiments
- `misc/`: Additional code for generating figures in our paper

#### Dependencies

SynDerm is relatively light-weight and depends primarily upon `diffusers` for image generation and `transformers` for text encoding. We provide a complete environment for reproducibility in `environment.yml`, but any recent version of these packages should work.

#### Dataset Structure

We provide code for loading the Fitzpatrick17k and DDI datasets in `generation/dataset.py`. The datasets are stored in a CSV file with the following information: the relative path to the image from some root directory, the skin condition label, and the skin tone classification. 

For example, for the Fitzpatrick17k dataset, the CSV file contains the following columns (as well as some additional columns which are not used here):
- `md5hash`: Unique identifier for each image
- `fitzpatrick_scale`: Skin tone classifications
- `label`: Specific skin condition diagnosis

To use a custom dataset, ensure it follows a similar structure and then modify `generation/dataset.py` (or create a new dataset class) to load your dataset.

#### Dreambooth Finetuning

Dreambooth finetuning is an optional step that we find significantly improves performance in limited data regimes. The code for finetuning is located in `generation/train_dreambooth.py` and `generation/train_dreambooth_inpaint.py`. After installing `synderm`, it can easily be accessed via the command line as follows: 








The Fitzpatrick-17k dataset class is located in `dataset.py`.

The code for finetuning the diffusion models is located in `train_dreambooth.py` and `train_dreambooth_inpaint.py`. For example, to finetune a model on basal cell carcinoma images, one may run:

```bash
python train_dreambooth.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
    --instance_prompt="An image of {}, a skin disease" \
    --validation_prompt="An image of basal cell carcinoma, a skin disease" \
    --output_dir=${OUTPUT_DIR} \
    --disease_class=basal-cell-carcinoma \
    --instance_data_dir=${FITZPATRICK17K_DATASET_DIR} \
    --resolution=512 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-06 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_train_epochs=4 \
    --report_to="wandb"
```

The generation code is located in `generate.py`. For example, to sample from a model, one may run:
```bash
python generate.py \
    --instance_data_dir=${FITZPATRICK17K_DATASET_DIR} \
    --model_type "text-to-image" \
    --pretrained_model_name_or_path=${PATH_TO_YOUR_PRETRAINED_OR_FINETUNED_MODEL} \
    --instance_prompt="An image of {}, a skin disease" \
    --disease_class=basal-cell-carcinoma
```

A full set of example scripts is provided in `scripts/examples.sh`.

### Experiments

Code for experiments is located in `experiments/`. Code is extensively commented. In each script, results are saved to Weights and Biases; if you have not already installed WandB, you can install it with `pip install wandb`. 
