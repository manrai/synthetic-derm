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
    "A clinical photograph of the same skin disease, {condition}, on a different patient who has the "
    "same skin tone, at a different body site, keeping the characteristic appearance of the lesions. "
    "No text, watermarks or annotations."
)

#: FLUX 3 addresses the reference image as <ref_image_0>.
FLUX_EDIT_TEMPLATE = (
    "Using <ref_image_0> as the reference, a clinical photograph of the same skin disease, {condition}, "
    "on a different patient who has the same skin tone, at a different body site, keeping the "
    "characteristic appearance of the lesions. No text, watermarks or annotations."
)


FITZPATRICK_PHRASES = {
    1: "very fair skin (Fitzpatrick type I)", 2: "fair skin (Fitzpatrick type II)",
    3: "light brown skin (Fitzpatrick type III)", 4: "medium brown skin (Fitzpatrick type IV)",
    5: "dark brown skin (Fitzpatrick type V)", 6: "deeply pigmented dark skin (Fitzpatrick type VI)",
}


def caption(label: str, template: str = PAPER_TEMPLATE, fitzpatrick: int | None = None) -> str:
    """Prompt for ``label``. ``fitzpatrick`` (1 to 6) appends the patient's skin tone.

    State the skin tone whenever the generator or its prompt upsampler is free to choose: left
    unspecified, FLUX 3 Klein's upsampler wrote "a white adult ... with fair skin" into every
    caption in the zero-shot vignette, which would skew a synthetic dataset toward light skin.
    """
    text = template.format(condition=label_words(label))
    if fitzpatrick is not None:
        if fitzpatrick not in FITZPATRICK_PHRASES:
            raise ValueError("fitzpatrick must be 1 to 6")
        text = text.rstrip(".") + f", on a patient with {FITZPATRICK_PHRASES[fitzpatrick]}."
    return text
