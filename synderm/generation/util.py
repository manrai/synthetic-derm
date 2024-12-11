import torch
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torchvision import transforms

class GenerationWrapper(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        instance_prompt: str,
        label_filter=None,
        size=512,
        center_crop=False
    ):
        self.dataset = dataset
        self.label_filter = label_filter

        # Filter the dataset to only include samples with the desired label
        self.filtered_indices = []
        if self.label_filter is not None:
            for i in range(len(self.dataset)):
                test_label = self.dataset[i]["label"]
                if test_label == self.label_filter:
                    self.filtered_indices.append(i)
        else:
            self.filtered_indices = list(range(len(self.dataset)))

        self.instance_prompt = instance_prompt

        self.num_instance_images = len(self.filtered_indices)
        self._length = self.num_instance_images

        image_transforms = [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
        ]
        normalize_and_to_tensor = [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        self.image_transforms = transforms.Compose(image_transforms + normalize_and_to_tensor)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}

        index_filtered = self.filtered_indices[index]

        # Access the data from the underlying dataset
        sample = self.dataset[index_filtered]
        image = sample["image"]
        label = sample["label"]

        # Prepare the instance prompt
        instance_class_name = label.replace('-', ' ')
        instance_prompt = self.instance_prompt.format(instance_class_name)

        # Apply image transformations
        if not image.mode == "RGB":
            image = image.convert("RGB")
        example["image"] = self.image_transforms(image)
        example["prompt"] = instance_prompt

        example["label"] = label
        example["id"] = sample["id"]

        return example

