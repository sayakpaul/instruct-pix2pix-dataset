"""
Adapted from https://github.com/timothybrooks/instruct-pix2pix/blob/main/edit_dataset.py
"""

import random

random.seed(0)

import json
from pathlib import Path
from random import shuffle

import torch
from PIL import Image
from torch.utils.data import Dataset


class EditDataset(Dataset):
    def __init__(self, path: str, num_samples_to_use: int, return_paths: bool = True):
        self.path = path

        with open(Path(self.path, "seeds.json")) as f:
            seeds = json.load(f)
        shuffle(seeds)
        if num_samples_to_use is not None:
            self.seeds = seeds[:num_samples_to_use]
        else:
            self.seeds = seeds

        self.return_paths = return_paths

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int) -> dict:
        name, seeds = self.seeds[i]
        prompt_dir = Path(self.path, name)
        seed = seeds[torch.randint(0, len(seeds), ()).item()]
        with open(prompt_dir.joinpath("prompt.json")) as fp:
            json_contents = json.load(fp)
            print(json_contents)
            edit_prompt = json_contents["edit"]
            original_prompt = json_contents["input"]
            edited_prompt = json_contents["output"]
            url = json_contents["url"]

        image_0_path = prompt_dir.joinpath(f"{seed}_0.jpg")
        image_1_path = prompt_dir.joinpath(f"{seed}_1.jpg")

        if self.return_paths:
            return dict(
                image_url=url,
                original_image=image_0_path,
                original_prompt=original_prompt,
                edit_prompt=edit_prompt,
                edited_prompt=edited_prompt,
                edited_image=image_1_path,
            )

        image_0 = Image.open(image_0_path).convert("RGB")
        image_1 = Image.open(image_1_path).convert("RGB")

        return dict(
            image_url=url,
            original_image=image_0,
            original_prompt=original_prompt,
            edit_prompt=edit_prompt,
            edited_prompt=edited_prompt,
            edited_image=image_1,
        )
