from synderm.generation.generate import generate_synthetic_dataset
from synderm.examples.sample_datasets import CustomDataset

from pathlib import Path

# TODO: make sure that this script works with both a pretrained model, and the fine-tuned model
sample_dataset = CustomDataset(dataset_dir="sample_dataset", split="train")

generate_synthetic_dataset(
    output_dir_path = Path("test_outputs"),
    generation_type = "inpaint", 
    model_path = "runwayml/stable-diffusion-inpainting",
    input_prompt = "An image of {}, a skin disease",
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

