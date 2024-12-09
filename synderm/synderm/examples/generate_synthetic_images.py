from synderm.generation.generate import generate_synthetic_dataset
from synderm.examples.sample_datasets import SampleDataset

from pathlib import Path

# TODO: make sure that this script works with both a pretrained model, and the fine-tuned model
sample_dataset = SampleDataset(dataset_dir="sample_dataset", split="train")

fine_tuned_model_path = "/n/scratch/users/t/thb286/dreambooth-outputs/allergic-contact-dermatitis"
#pretrained_model_path = "runwayml/stable-diffusion-inpainting"

output_dir = Path("test_outputs")

generate_synthetic_dataset(
    label_filter="allergic-contact-dermatitis",
    dataset=sample_dataset,
    output_dir_path = output_dir,
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

