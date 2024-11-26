from datasets import load_dataset, Image
from synderm.models.diffusion_model import DiffusionModel
from synderm.synderm.generation.synthetic_generation import generate_synthetic_images
from synderm.splits.train_test_splitter import create_train_test_split

# Load and preprocess data
dataset = load_dataset("beans", split="train")

# Initialize and train the model
model = DiffusionModel(config={"base_model": "stabilityai/stable-diffusion-2-1-base"})
# python train_dreambooth.py --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" --instance_data_dir=/home/lukemelas/data/Fitzpatrick17k --instance_prompt="An image of {}, a skin disease" --validation_prompt="An image of allergic contact dermatitis, a skin disease" --output_dir="dreambooth-outputs/allergic-contact-dermatitis" --disease_class=allergic-contact-dermatitis --resolution=512 --train_batch_size=4 --gradient_accumulation_steps=1 --learning_rate=5e-06 --lr_scheduler="constant" --lr_warmup_steps=0 --num_train_epochs=4 --report_to="wandb"

# Model has to be trained for each class

# Optional -- if we don't call this we will use the pretrained model
model.train(
    dataset, 
    labels = {"allergic-contact-dermatitis"},  # we can select a subset of the labels if we'd like
    n_gpus = 1,
    output_directory = "models_test" # as models are trained, saves them to this output directory with a subdirectory for each model label
    )

# ...

# Generate synthetic images
synthetic_images = generate_synthetic_images(
    dataset, 
    #model_path = "stabilityai/stable-diffusion-2-1-base", 
    model_path = "models_test",   # must contain a model for each label in the dataset if this option is chosen
    num_images=1000, 
    method="text-to-image")

# python generate.py --output_root generations-pretrained --instance_data_dir=${FITZPATRICK17K_DATASET_DIR} --model_type "text-to-image" --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" --instance_prompt="An image of {}, a skin disease" --disease_class=allergic-contact-dermatitis

# Create train-test splits
train_data, test_data = create_train_test_split(data, synthetic_images, ratio=0.8)

# 
