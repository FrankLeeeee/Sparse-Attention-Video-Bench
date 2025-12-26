from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
import argparse

BENCHMARK_DIR = Path(__file__).parent
VBENCH_DIRECTORY = BENCHMARK_DIR.joinpath("VBench")
INSTRUCTION = "Can you help me refine the following video caption for a video generation task? The caption is: {caption}. Please answer only with one sentence."
DIMENSIONS = ["appearance_style", "color", "human_action", "multiple_objects", "object_class", "overall_consistency", "scene", 'spatial_relationship', 'subject_consistency', 'temporal_style', 'temporal_flickering']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dimension", type=str, choices=DIMENSIONS + ["all"], required=True)
    return parser.parse_args()

def fetch_vbench_prompts(vbench_directory: str, dimension: str):
    vbench_dir = Path(vbench_directory)
    prompts_dir = vbench_dir.joinpath("prompts", "prompts_per_dimension")

    # find all txt files 
    txt_file = prompts_dir.joinpath(f"{dimension}.txt")

    with open(txt_file, "r") as f:
        prompts = f.readlines()

    return prompts


def main():
    args = parse_args()
    client = OpenAI()
    prompts = fetch_vbench_prompts(VBENCH_DIRECTORY, args.dimension)

    optimized_prompts = []
    for prompt in tqdm(prompts, desc=f"Processing {args.dimension}"):
            response = client.responses.create(
                model="gpt-5-mini",
                input=INSTRUCTION.format(caption=prompt)
            )
            print(response)
            optimized_prompts.append(response.output_text)

    # save the data as a json file
    with open(BENCHMARK_DIR.joinpath("prompts", f"optimized_{args.dimension}.txt"), "w") as f:
        for prompt in optimized_prompts:
            f.write(prompt + "\n")

if __name__ == "__main__":
    main()
