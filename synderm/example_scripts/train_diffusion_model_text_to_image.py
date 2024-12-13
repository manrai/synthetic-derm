from synderm.fine_tune.text_to_image_diffusion import fine_tune_text_to_image
from synderm.example_scripts.sample_datasets import SampleDataset, FitzDataset

#sample_dataset = SampleDataset(dataset_dir="sample_derm_dataset", split="train")
sample_dataset = FitzDataset(
    images_path="/n/data1/hms/dbmi/manrai/derm/Fitzpatrick17k/finalfitz17k",
    metadata_path="fitz_metadata/fitzpatrick17k_10label_clean_training.csv"
    )

fine_tune_text_to_image(
    train_dataset=sample_dataset,
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-base",
    instance_prompt = "An image of {}, a skin disease",
    validation_prompt_format = "An image of {}, a skin disease",
    output_dir = "/n/scratch/users/t/thb286/dreambooth-outputs-text-to-image",
    label_filter = "allergic-contact-dermatitis",
    resolution = 512,
    train_batch_size = 4,
    gradient_accumulation_steps = 1,
    learning_rate = 5e-6,
    lr_scheduler = "constant",
    lr_warmup_steps = 0,
    num_train_epochs = 4,
    report_to = "wandb"
)
