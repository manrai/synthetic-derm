synderm documentation
=====================

Synderm generates synthetic dermatology images with modern diffusion models and mixes them into
classifier training. Version 0.2 supports three generator backends (Qwen-Image-2.1, FLUX 3 Image
Klein and SDXL), trains LoRA adapters through the upstream diffusers scripts, and trains
classifiers with timm. It accompanies the paper *Augmenting medical image classifiers with
synthetic data from latent diffusion models* and the 1M-image synthetic dataset on Hugging Face.

Vignettes run end to end on one 32 GB GPU against real Fitzpatrick 17k images that are fetched
from Atlas Dermatologico at runtime (see :mod:`synderm.data`).

.. toctree::
   :maxdepth: 1
   :caption: Vignettes

   examples/00_public_dataset.ipynb
   examples/01_zero_shot_generation.ipynb
   examples/02_finetune_lora.ipynb
   examples/03_image_conditioned_augmentation.ipynb
   examples/04_augment_with_public_dataset.ipynb
   examples/05_low_data_benchmark.ipynb
   examples/06_synthetic_image_quality.ipynb
   examples/07_performance_by_skin_tone.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Reference

   models
   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
