# Ethics and licences

## Ethics
- Reconstructing a face from a voice carries privacy and misuse risks. Generated faces are statistical guesses, not
  identifications, and must not be used as evidence of who someone is. The measured retrieval rates are low.
- Use only data you are permitted to use, and do not publish generated faces of real, named people.
- Demographic subgroup analysis uses only labels that ship with the dataset (for example sex in `vox2_meta.csv`). Do not infer protected attributes from faces or voices.

## Third-party licences (check before use, especially commercial use)
- **VoxCeleb2**: academic terms on the official page; you must request access and may not redistribute the data.
- **Arc2Face and its model weights**, **insightface models**, **Stable Diffusion 1.5**, **SpeechBrain ECAPA**: each has its own licence,
  and some are non-commercial. Read the licence of every model you download.
- This repository's own code is MIT (see `LICENSE`).
