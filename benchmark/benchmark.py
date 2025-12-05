import argparse
import json
from pathlib import Path
import pandas as pd

import yaml
from loguru import logger
from models.model_types import Models


def main():
    parser = argparse.ArgumentParser(description="Run Baseline Models")

    parser.add_argument(
        "-c", "--config_file",
        type=str,
        required=True,
        help="Path to the configuration file"
    )
    parser.add_argument(
        "-i", "--input_file",
        type=str,
        required=True,
        help="Path to the input file"
    )
    parser.add_argument(
        "-o", "--output_file",
        type=str,
        required=False,
        help="Path to the output file"
    )
    parser.add_argument("--device_map",
                        type=str,
                        default="cuda")

    parser.add_argument("--no_context",
                        type=bool,
                        default=False)

    parser.add_argument("-cot", "--chain_of_thought",
                        type=bool,
                        default=False)

    # NOTE: Path to few-shot exemplars is set in the config file
    parser.add_argument("-fws", "--few_shot",
                        type=bool,
                        default=False)

    # NOTE: We intend to do a more intuitive loading via HF in a future version
    parser.add_argument("-mmlm", "--multimodal_file",
                        type=str,
                        default="",
                        help="Path to the Arrow-based Image Store")

    args = parser.parse_args()

    # Load the configuration file
    logger.info(f"Loading the YAML configuration file `{args.config_file}`...")
    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    # File paths
    input_file = args.input_file
    output_file = args.output_file
    device_map = args.device_map
    if device_map.isnumeric():
        device_map = int(device_map)

    if not output_file:
        output_dir = Path(__file__).resolve().parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        file_name = Path(args.config_file).stem
        output_file = output_dir / f"{file_name}.jsonl"

    # Load the input file
    logger.info(f"Loading the input file `{input_file}`...")

    # At present - we pass path to input file; we intend to add support for direct HF download
    input_rows = pd.read_parquet(input_file)

    # Load the model
    m = Models.get_model(config["model"], args.multimodal)
    model = m(config, device_map, args.multimodal)

    # Generate predictions
    results = model.generate_predictions(input_rows, args.no_context, args.chain_of_thought,
                                         args.few_shot, args.multimodal)

    # Save the results
    logger.info(f"Saving the results to `{output_file}`...")
    with open(output_file, "w+") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    logger.info("Done!")


if __name__ == "__main__":
    main()
