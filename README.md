This repository provides utilities to a minimal dataset for [InstructPix2Pix](https://arxiv.org/abs/2211.09800) like training for Diffusion models.

## Steps

1. Download the original dataset as discussed [here](https://github.com/timothybrooks/instruct-pix2pix#generated-dataset). I used this version: `clip-filtered-dataset`. Note that the download can take as long as 24 hours depending on the internet bandwidth. The dataset also requires at least 600 GB of storage.
2. Then run:

    ```bash
    python make_dataset.py --data_root clip-filtered-dataset --num_samples_to_use 1000
    ```
3. The `make_dataset.py` was specifically designed to obtain a [🤗 dataset](https://huggingface.co/docs/datasets/). So, it's the most useful when you push the minimal dataset to the 🤗 Hub. You can do so by setting `push_to_hub` while running `make_dataset.py`. 

## Example dataset

