#from synderm.generation.generate import generate_synthetic_dataset

from synderm.generation.generate import generate_synthetic_dataset
from synderm.examples.sample_datasets import SampleDataset, FitzDataset

#sample_dataset = SampleDataset(dataset_dir="sample_derm_dataset", split="train")
sample_dataset = FitzDataset(
    images_path="/n/data1/hms/dbmi/manrai/derm/Fitzpatrick17k/finalfitz17k",
    metadata_path="fitz_metadata/fitzpatrick17k_10label_clean_training.csv"
    )

fine_tuned_model_path = "/n/scratch/users/t/thb286/dreambooth-outputs/allergic-contact-dermatitis"
#pretrained_model_path = "runwayml/stable-diffusion-inpainting"

generate_synthetic_dataset(
    dataset=sample_dataset,
    label_filter="allergic-contact-dermatitis",
    output_dir_path = "/n/scratch/users/t/thb286/generations",
    generation_type = "text-to-image", 
    model_path = fine_tuned_model_path,
    instance_prompt = "An image of {}, a skin disease",
    batch_size = 16,
    start_index = 0,
    num_generations_per_image = 1,
    seed = 42,
    guidance_scale = 3.0,
    num_inference_steps = 50,
    strength_inpaint = 0.970,
    strength_outpaint = 0.950,
    mask_fraction = 0.25
)

