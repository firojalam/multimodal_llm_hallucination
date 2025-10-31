import argparse
import json
import logging
import os

from dotenv import load_dotenv
from generate_QA_batch_correct import (
    AzureOpenAIBatchManager,
    get_dictionary_keys,
    ImageBatchProcessor,
)
from tqdm import tqdm


# Function to configure logging
def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


# Function to load environment variables
def load_env(env_path):
    load_dotenv(dotenv_path=env_path, override=True)
    deployment_name = os.environ["AZURE_ENGINE_NAME"]
    openai_api_base = os.environ["AZURE_API_URL"]
    openai_api_key = os.environ["AZURE_API_KEY"]
    openai_api_version = os.environ["AZURE_API_VERSION"]
    api_url = openai_api_base

    headers = {"api-key": openai_api_key}
    return openai_api_key, api_url, openai_api_version, headers, deployment_name


# Function to remove duplicates from a JSONL file
def remove_duplicates_from_jsonl(file_path):
    unique_custom_ids = set()
    cleaned_lines = []

    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            if "custom_id" in data:
                custom_id = data["custom_id"]
                if custom_id not in unique_custom_ids:
                    unique_custom_ids.add(custom_id)
                    cleaned_lines.append(line)
            else:
                logging.warning("Line missing 'custom_id': %s", line.strip())

    with open(file_path, "w") as file:
        for line in cleaned_lines:
            file.write(line)


# Iterate through all JSONL files in the directory and remove duplicates
def remove_duplicates(output_dir):
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".jsonl"):
            file_path = os.path.join(output_dir, file_name)
            remove_duplicates_from_jsonl(file_path)
            logging.info(f"Processed {file_name} to remove duplicates.")


# Main function
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate QA batch submission and remove duplicates."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./images",
        help="Input file or source directory path",
    )
    parser.add_argument(
        "--env_file", type=str, required=True, help="Path to the environment file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cached_dir/QA/Qatar/GPT-4o_batch/",
        help="Output directory path",
    )
    parser.add_argument(
        "--batch_file",
        type=str,
        default="cached_dir/QA/Qatar/batch_files.txt",
        help="File to store batch file names",
    )
    args = parser.parse_args()

    # Configure logging
    configure_logging()

    # Resolve paths
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    env_path = args.env_file
    if not os.path.exists(env_path):
        logging.error(f"Error: {env_path} not found!")
        return

    # Load environment variables
    logging.info("Loading environment variables...")
    (
        azure_openai_api_key,
        azure_openai_api_url,
        openai_api_version,
        _,
        deployment_name,
    ) = load_env(env_path)

    intermediate_batch_file_name = args.batch_file
    input_source = args.input
    batch_processor = ImageBatchProcessor(input_source, output_dir, deployment_name)
    batch_processor.create_batches()

    # Remove duplicates from JSONL files
    logging.info("Removing duplicates from JSONL files...")
    remove_duplicates(output_dir)

    batch_manager = AzureOpenAIBatchManager(
        azure_openai_api_key,
        azure_openai_api_url,
        openai_api_version,
        deployment_name=deployment_name,
        batch_file_name=intermediate_batch_file_name,
        # output_dir=output_dir,
    )

    batch_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".jsonl")
    ]
    print("Number of batch files:", len(batch_files))

    # Submit all tasks in a directory and save their batch IDs ===
    batch_manager.submit_all_batches_in_directory(output_dir, verbose=True)

    logging.info("Process completed successfully.")


if __name__ == "__main__":
    main()
