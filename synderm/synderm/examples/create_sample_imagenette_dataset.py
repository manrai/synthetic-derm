import os
import shutil

source_dir = "imagenette2"
subset_dir = "imagenette2_subset"

label_map = {
    "n01440764": "tench",
    "n02102040": "English_springer",
    "n02979186": "cassette_player",
    "n03000684": "chain_saw",
    "n03028079": "church",
    "n03394916": "French_horn",
    "n03417042": "garbage_truck",
    "n03425413": "gas_pump",
    "n03445777": "golf_ball",
    "n03888257": "parachute"
}

# Define per-class training limits
train_limits = {
    "n02102040": 20  # English_springer
}
default_train_limit = 200

"""
Creates a subset of the Imagenette2 dataset with a specified number of training images per class
while keeping the validation set intact. Selects only the first 4 classes and renames them based on label_map.
The "English_springer" class is limited to 20 training images, while other classes have 200 training images each.
"""

# Define paths for train and validation directories
train_src = os.path.join(source_dir, 'train')
val_src = os.path.join(source_dir, 'val')

train_dst = os.path.join(subset_dir, 'train')
val_dst = os.path.join(subset_dir, 'val')

# Create destination directories if they don't exist
os.makedirs(train_dst, exist_ok=True)
os.makedirs(val_dst, exist_ok=True)

# Get list of all class directories and select first 4
all_classes = [d for d in os.listdir(train_src) if os.path.isdir(os.path.join(train_src, d))]

# Process training data
for class_name in all_classes:
    class_src = os.path.join(train_src, class_name)
    renamed_class = label_map.get(class_name, class_name)
    class_dst = os.path.join(train_dst, renamed_class)

    os.makedirs(class_dst, exist_ok=True)

    # List all JPEG images in the class directory
    images = [f for f in os.listdir(class_src) if f.lower().endswith('.jpeg') or f.lower().endswith('.jpg')]

    # Determine the train limit for the current class
    train_limit = train_limits.get(class_name, default_train_limit)

    # Select the first 'train_limit' images
    selected_images = images[:train_limit]

    for image in selected_images:
        src_image_path = os.path.join(class_src, image)
        dst_image_path = os.path.join(class_dst, image)
        shutil.copy2(src_image_path, dst_image_path)

    print(f"Copied {len(selected_images)} images for class '{renamed_class}' to the subset training set.")

# Process validation data (copy only the selected classes)
for class_name in all_classes:
    class_src = os.path.join(val_src, class_name)
    renamed_class = label_map.get(class_name, class_name)
    class_dst = os.path.join(val_dst, renamed_class)

    os.makedirs(class_dst, exist_ok=True)

    # List all JPEG images in the class directory
    images = [f for f in os.listdir(class_src) if f.lower().endswith('.jpeg') or f.lower().endswith('.jpg')]

    for image in images:
        src_image_path = os.path.join(class_src, image)
        dst_image_path = os.path.join(class_dst, image)
        shutil.copy2(src_image_path, dst_image_path)

    print(f"Copied {len(images)} images for class '{renamed_class}' to the subset validation set.")