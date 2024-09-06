# SynDerm: MVP Desiderata

## Installation
- [ ] pip install synderm

## Core Functionality
- [ ] Image generation (text-to-image, inpainting, outpainting, in-then-outpainting)
- [ ] Fine-tuning of pre-trained models (DreamBooth)
- [ ] Synthetic data augmentation for classifier training
- [ ] Auto-loading of ~1M pre-generated images released alongside the paper (on HuggingFace)

## Features
- [ ] Flexibility to work with user-defined skin condition labels
- [ ] Customizable synthetic-to-real image ratios
- [ ] PyTorch integration
- [ ] Experiment framework to assess utility of synthetic images
- [ ] Secure local processing of private datasets
- [ ] Public dataset support: Fitzpatrick 17k and Stanford DDI
- [ ] Vignette demonstrating full workflow

## Documentation
- [ ] README with quick start guide
- [ ] Jupyter notebook tutorial
- [ ] Vignette: "Augmenting a Skin Condition Classifier with Synthetic Data"

## Data Handling
- [ ] Load private datasets
- [ ] Auto-download and load Fitzpatrick 17k and Stanford DDI datasets
- [ ] Option to use pre-generated images or generate new ones

## Extensibility (?)
- [ ] Hooks for custom generation methods, classifiers, and evaluation metrics
