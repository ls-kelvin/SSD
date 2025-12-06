import argparse
import json
import os
import random


def generate_merged_validation_json(input_dir, output_file, num_elements=64, num_frames=77, height=480, width=832, num_inference_steps=40, seed=None):
    # read in video2caption.json
    with open(os.path.join(input_dir, "videos2caption.json"), "r") as f:
        video2caption = json.load(f)

    # count how many elements are in the list
    num_available = len(video2caption)
    print(f"Number of elements in video2caption.json: {num_available}")

    # set random seed (if provided)
    if seed is not None:
        random.seed(seed)

    # ensure sampled number of elements doesn't exceed available
    if num_elements > num_available:
        print(f"Requested num_elements ({num_elements}) > available ({num_available}), using {num_available} instead.")
        num_elements = num_available

    # randomly sample
    sampled_elements = random.sample(video2caption, num_elements)

    # Transform sampled elements into validation.json format
    validation_data = []
    for element in sampled_elements:
        assert element.get("cap") is not None, f"Caption is None for element: {element}"
        validation_entry = {
            "caption": element["cap"][0],
            "video_path": None,
            "num_inference_steps": num_inference_steps,
            "height": height,
            "width": width,
            "num_frames": num_frames,
        }
        if "action_path" in element:
            validation_entry["action_path"] = element["action_path"]
        validation_data.append(validation_entry)

    # Create the final validation structure
    validation_json = {"data": validation_data}

    # Write the validation JSON to the output file
    with open(output_file, "w") as f:
        json.dump(validation_json, f, indent=2)

    print(f"Generated validation JSON with {len(validation_data)} entries and saved to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    # dataset_type: "mixkit"
    parser.add_argument("--dataset_type", choices=["merged"], required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--num_elements", type=int, default=64, help="Number of samples to include in validation.json")
    parser.add_argument("--num_frames", type=int, default=77, help="Number of frames per sample")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--width", type=int, default=832, help="Frame width")
    parser.add_argument("--num_inference_steps", type=int, default=40, help="Inference steps for each sample")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible sampling")
    args = parser.parse_args()

    if args.dataset_type == "merged":
        generate_merged_validation_json(
            args.input_dir,
            args.output_file,
            num_elements=args.num_elements,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()