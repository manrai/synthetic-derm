import pandas as pd

def synthetic_train_val_split(
        real_data, 
        synthetic_data,
        test_size=None,
        per_class_size=40,
        n_real_per_class=None,
        n_synthetic_per_class=None,
        random_state=None,
        class_column="label",
        id_column="md5hash"):
    """
    Splits the combined dataset of real and synthetic images into training and validation sets.
    The training set consists of synthetic images and a limited number of real images per class.
    In the training set, only synthetic entries will only be included if they have a corresponding entry in the training set. So, there needs to be a mapping from real images to synthetic images (usually based on the way they are generated, for example inpaint synthetic images will have an ID corresponding to the source image). 
    The validation set consists only of real images.

    Parameters:
    -----------
    real_data : pandas.DataFrame
        DataFrame containing the real images data.
    synthetic_data : pandas.DataFrame
        DataFrame containing the synthetic images data.
    test_size : float, int, or None, default=None
        If float, represents the proportion of the real dataset to include in the validation split.
        If int, represents the total number of validation samples.
        If None, `per_class_size` must be specified.
    per_class_size : int or None, default=40
        Number of samples per class to include in the validation set.
        If None, `test_size` must be specified.
    n_real_per_class : int or None, default=None
        Maximum number of real images per class to include in the training data.
        If None, all remaining real images (excluding validation set) are included.
    random_state : int or None, default=None
        Random seed for reproducibility.
    class_column : str, default="label"
        Name of the column in the DataFrames that contains the class labels.
    id_column : str, default="md5hash"
        Name of the column in the DataFrames that contains unique identifiers.

    Returns:
    --------
    train_data : pandas.DataFrame
        DataFrame containing the training data.
    val_data : pandas.DataFrame
        DataFrame containing the validation data.
    """
    # Ensure the original dataframes are not modified
    real_data = real_data.copy()
    synthetic_data = synthetic_data.copy()

    # Assign 'synthetic' flag
    real_data["synthetic"] = False
    synthetic_data["synthetic"] = True

    # Validate input parameters
    if (test_size is None and per_class_size is None) or (test_size is not None and per_class_size is not None):
        raise ValueError("Specify exactly one of 'test_size' or 'per_class_size'.")

    val_data_list = []

    if per_class_size is not None:
        # Sample specified number per class for validation
        for class_name, group in real_data.groupby(class_column):
            n_samples = min(len(group), per_class_size)
            val_samples = group.sample(n=n_samples, random_state=random_state, replace=False)
            val_data_list.append(val_samples)
    else:
        # Compute per-class validation sizes based on test_size
        if isinstance(test_size, float):
            if not 0.0 < test_size < 1.0:
                raise ValueError("When 'test_size' is a float, it must be between 0.0 and 1.0.")
            for class_name, group in real_data.groupby(class_column):
                n_samples = max(1, int(round(len(group) * test_size)))
                val_samples = group.sample(n=n_samples, random_state=random_state, replace=False)
                val_data_list.append(val_samples)
        elif isinstance(test_size, int):
            total_samples = test_size
            num_classes = real_data[class_column].nunique()
            samples_per_class = total_samples // num_classes
            remainder = total_samples % num_classes
            for idx, (class_name, group) in enumerate(real_data.groupby(class_column)):
                n_samples = samples_per_class + (1 if idx < remainder else 0)
                n_samples = min(len(group), n_samples)
                val_samples = group.sample(n=n_samples, random_state=random_state, replace=False)
                val_data_list.append(val_samples)
        else:
            raise ValueError("'test_size' must be a float or int.")

    # Combine validation samples
    val_data = pd.concat(val_data_list).reset_index(drop=True)

    # Exclude validation samples from real_data
    val_ids = val_data[id_column].unique()
    remaining_real_data = real_data[~real_data[id_column].isin(val_ids)]

    # Create training data
    if n_real_per_class is not None:
        # Limit the number of real images per class in training data
        train_real_data_list = []
        for class_name, group in remaining_real_data.groupby(class_column):
            n_samples = min(len(group), n_real_per_class)
            train_samples = group.sample(n=n_samples, random_state=random_state, replace=False)
            train_real_data_list.append(train_samples)
        train_real_data = pd.concat(train_real_data_list).reset_index(drop=True)
    else:
        # Use all remaining real images
        train_real_data = remaining_real_data.copy()
    
    if n_synthetic_per_class is not None:
        raise NotImplementedError()

    # Get the synthetic images that share the same md5hash as real images in the training data
    md5_train = train_real_data['md5hash']
    synthetic_data_subset = synthetic_data[synthetic_data['md5hash'].isin(md5_train)]

    # Combine synthetic data with the sampled real data for training
    train_data = pd.concat([train_real_data, synthetic_data_subset]).reset_index(drop=True)

    # Add 'is_valid' column
    train_data['is_valid'] = False
    val_data['is_valid'] = True

    return train_data, val_data
