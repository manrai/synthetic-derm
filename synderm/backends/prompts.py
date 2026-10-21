"""Prompt templates. ``{condition}`` is replaced by the label with hyphens turned into spaces."""
from synderm.data.datasets import label_words

#: The prompt used in the 2023 paper for every generation method.
PAPER_TEMPLATE = "An image of {condition}, a skin disease"

#: A fuller caption for models that expect scene descriptions (FLUX 3 Klein without its upsampler).
CLINICAL_TEMPLATE = (
    "A clinical dermatology photograph of {condition}, a skin disease, showing the affected skin "
    "in close-up under even lighting, sharp focus, no text or annotations."
)

#: Instruction for reference-conditioned editing: same condition, different patient.
EDIT_TEMPLATE = (
    "A clinical photograph of the same skin disease, {condition}, on a different patient and a "
    "different body site, keeping the characteristic appearance of the lesions."
)

#: FLUX 3 addresses the reference image as <ref_image_0>.
FLUX_EDIT_TEMPLATE = (
    "Using <ref_image_0> as the reference, a clinical photograph of the same skin disease, {condition}, "
    "on a different patient and a different body site, keeping the characteristic appearance of the lesions."
)


def caption(label: str, template: str = PAPER_TEMPLATE) -> str:
    return template.format(condition=label_words(label))
