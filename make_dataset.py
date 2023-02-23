import argparse

from datasets import Dataset, Features
from datasets import Image as ImageFeature
from datasets import Value

from minimal_edit_dataset import EditDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a mini dataset fro InstructPix2Pix style training."
    )
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--num_samples_to_use", type=int, default=1000)
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()
    return args


def gen_examples(dataset):
    def fn():
        for sample in dataset:
            print(f"From generator fn: {sample}")
            yield {
                "input_image": {"path": sample["input_image"]},
                "edit_prompt": sample["edit_prompt"],
                "edited_image": {"path": sample["edited"]},
            }

    return fn


def main(args):
    mini_edit_dataset = EditDataset(args.data_root, args.num_samples_to_use)
    print(f"Total samples: {len(mini_edit_dataset)}")
    generator_fn = gen_examples(mini_edit_dataset)

    print("Creating dataset...")
    mini_ds = Dataset.from_generator(
        generator_fn,
        features=Features(
            input_image=ImageFeature(),
            edit_prompt=Value("string"),
            edited_image=ImageFeature(),
        ),
    )

    if args.push_to_hub:
        print("Pushing to the Hub...")
        num_samples = args.num_samples_to_use
        ds_name = f"instructpix2pix-{num_samples}-samples"
        mini_ds.push_to_hub(ds_name)

if __name__ == "__main__":
    args = parse_args()
    main(args)