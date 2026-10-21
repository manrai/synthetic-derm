"""synderm: synthetic dermatology images for augmenting image classifiers.

Subpackages are imported lazily so that light-weight use (data access, classifier training)
does not import diffusers.

- :mod:`synderm.data`       datasets, the Atlas Dermatologico vignette sample, training-data export
- :mod:`synderm.backends`   generator backends: SDXL, Qwen-Image-2.1, FLUX 3 Image Klein
- :mod:`synderm.fine_tune`  LoRA fine-tuning through the upstream diffusers DreamBooth scripts
- :mod:`synderm.generation` bulk synthetic image generation with per-image metadata
- :mod:`synderm.classify`   timm classifier training and evaluation
- :mod:`synderm.eval`       synthetic image quality metrics and reader-study export
- :mod:`synderm.utils`      real/synthetic train-validation splitting
"""
__version__ = "0.2.0.dev0"
__all__ = ["__version__"]
