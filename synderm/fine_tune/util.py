from functools import lru_cache
import torch
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torchvision import transforms

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    from transformers import PretrainedConfig

    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

def collate_fn(examples, with_prior_preservation=False):
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

        if has_attention_mask:
            attention_mask += [example["class_attention_mask"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }

    if has_attention_mask:
        attention_mask = torch.cat(attention_mask, dim=0)
        batch["attention_mask"] = attention_mask

    if "prompt" in examples[0]:
        batch["prompt"] = [example["prompt"] for example in examples]

    return batch

class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

class DiffusionTrainWrapper(Dataset):
    def __init__(
        self,
        train_dataset: Dataset,
        tokenizer,
        instance_prompt: str,
        label_filter=None,
        size=512,
        center_crop=False,
        encoder_hidden_states=None,
        instance_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
        add_fitzpatrick_scale_to_prompt: bool = False
    ):
        self.train_dataset = train_dataset
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.instance_prompt_encoder_hidden_states = instance_prompt_encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length
        self.label_filter = label_filter

        # Filter the dataset to only include samples with the desired label
        self.filtered_indices = []
        if self.label_filter is not None:
            for i in range(len(train_dataset)):
                test_label = self.train_dataset[i]["label"]
                if test_label == self.label_filter:
                    self.filtered_indices.append(i)
        else:
            self.filtered_indices = list(range(len(train_dataset)))

        self.num_instance_images = len(self.filtered_indices)
        self.add_fitzpatrick_scale_to_prompt = add_fitzpatrick_scale_to_prompt

        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        # Image transforms
        image_transforms = [
            transforms.RandomResizedCrop(
                size=size, scale=(0.9, 1.1), ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
            ),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.RandomHorizontalFlip(),
        ]

        normalize_and_to_tensor = [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        self.image_transforms = transforms.Compose(image_transforms + normalize_and_to_tensor)

    def __len__(self):
        return self._length

    @lru_cache()
    def tokenize_prompt_with_caching(self, instance_prompt: str):
        return tokenize_prompt(self.tokenizer, instance_prompt, tokenizer_max_length=self.tokenizer_max_length)

    def __getitem__(self, index):
        example = {}

        index_filtered = self.filtered_indices[index]

        # Access the data from the underlying train_dataset
        train_sample = self.train_dataset[index_filtered]
        image = train_sample["image"]
        label = train_sample["label"]

        # Prepare the instance prompt
        instance_class_name = label.replace('-', ' ')
        instance_prompt = self.instance_prompt.format(instance_class_name)
        if self.add_fitzpatrick_scale_to_prompt and hasattr(train_sample, "fitzpatrick_scale"):
            fitzpatrick_scale = str(train_sample.fitzpatrick_scale) if train_sample.fitzpatrick_scale > 0 else 'unknown'
            instance_prompt += f', Fitzpatrick skin type {fitzpatrick_scale}'

        # Apply image transformations
        if not image.mode == "RGB":
            image = image.convert("RGB")
        example["instance_images"] = self.image_transforms(image)

        # Tokenize the prompt with caching
        if self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            text_inputs = self.tokenize_prompt_with_caching(instance_prompt)
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask
            example["prompt"] = instance_prompt

        return example

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs

def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds
