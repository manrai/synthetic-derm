Supported generator models
==========================

.. list-table::
   :header-rows: 1

   * - Backend name
     - Model
     - Text-to-image
     - Reference-image edit
     - Hard-mask inpaint
     - LoRA training
     - License
   * - ``qwen-image-2.1``
     - `Qwen/Qwen-Image-2.1 <https://huggingface.co/Qwen/Qwen-Image-2.1>`_ (7B)
     - yes
     - yes
     - no
     - text-to-image, image-to-image
     - Qwen Research (research only)
   * - ``flux-3-klein``
     - `black-forest-labs/flux-3-image-klein <https://huggingface.co/black-forest-labs/flux-3-image-klein>`_ (5.6B; ``-base`` for training)
     - yes
     - yes
     - no
     - text-to-image, image-to-image
     - FLUX Kommunity (non-commercial)
   * - ``sdxl``
     - `stabilityai/stable-diffusion-xl-base-1.0 <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0>`_ plus the SDXL inpainting checkpoint
     - yes
     - no
     - yes (inpaint, in-then-outpaint)
     - text-to-image
     - CreativeML OpenRAIL++-M

The 2023 paper used Stable Diffusion 2.1-base and SD 1.5 inpainting; that code lives unchanged
in ``paper_experiments/``. See ``MODERNIZATION_PLAN.md`` in the repository for how the three
backends were chosen.
