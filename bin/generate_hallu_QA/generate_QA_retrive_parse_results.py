import argparse
import json
import logging
import os

import pandas as pd
from dotenv import load_dotenv
from generate_QA_batch import (
    AzureOpenAIBatchManager,
    get_dictionary_keys,
    ImageBatchProcessor,
)
from tqdm import tqdm


key_variations = {
    "description": ["description", "Description"],
    "extracted_text": ["extracted_text", "Extracted_text", "Text", "text"],
    "image_category": ["image_category", "Image_category", "Category", "category"],
    "status": ["status", "Status", "Suitable", "suitable"],
    "reason": ["reason", "Reason", "Explanation", "explanation"],
}


# Function to configure logging
def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def load_env(env_path):
    load_dotenv(dotenv_path=env_path, override=True)
    deployment_name = os.environ["AZURE_ENGINE_NAME"]
    openai_api_base = os.environ["AZURE_API_URL"]
    openai_api_key = os.environ["AZURE_API_KEY"]
    openai_api_version = os.environ["AZURE_API_VERSION"]
    api_url = openai_api_base

    headers = {"api-key": openai_api_key}
    return openai_api_key, api_url, openai_api_version, headers


def post_processing_batch_results(file_path):
    """
    Read the content from the jsonl file and convert each line to a dictionary,
    then return a DataFrame with the relevant information.

    Args:
        file_path (str): The full path to the jsonl file.

    Returns:
        data: A list of dictionaries containing the relevant information.
    """
    with open(file_path, "r") as f:
        file_content = f.readlines()

    data = {}
    for line in file_content:
        try:
            t_file_content_dict = json.loads(line)
            t_file_name_original = os.path.splitext(t_file_content_dict["custom_id"])[0]
            if "response" not in t_file_content_dict:
                continue

            model = t_file_content_dict["response"]["body"]["model"]
            t_response = json.loads(
                t_file_content_dict["response"]["body"]["choices"][0]["message"][
                    "content"
                ]
            )
            data[t_file_name_original] = {
                "id": t_file_name_original,
                "response": t_response,
                "model": model,
            }
        except Exception as e:
            logging.error(
                f"Error in parsing the response for {file_path}, {t_file_name_original}"
            )
            logging.error(f"Error: {e}")
            continue
    logging.info(
        f"Number of data points {len(file_content)} and successfully parsed: {len(data)}"
    )
    return data


def process_batch_error_data(file_list, original_file, output_file):
    all_batch_data = []
    for file_path in tqdm(file_list):
        logging.info(f"Retrived error file to process: {file_path}")
        with open(file_path, "r") as f:
            file_content = f.readlines()
        for line in file_content:
            try:
                json_object = json.loads(line)
                image_id = os.path.splitext(os.path.basename(json_object["custom_id"]))[
                    0
                ]
                all_batch_data.append(image_id)
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON in file {file_path}: {e}")
                continue

    logging.info("Number of data points in batch: {}".format(len(all_batch_data)))
    all_original_data = {}
    with open(original_file, "r") as f:
        non_photograph_image_category_count = 0
        for line in f:
            data = json.loads(line)
            image_id = data["image_id"]
            if image_id in all_batch_data:
                all_original_data[image_id] = data

    logging.info(
        "Number of data points with original: {}".format(len(all_original_data))
    )
    logging.info(
        f"Number of non-photograph image categories: {non_photograph_image_category_count}"
    )
    with open(output_file, "w") as f:
        for image_id, data in all_original_data.items():
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    logging.info(f"Output saved to: {output_file}")


def process_batch_data(file_list, original_file, output_file):
    all_batch_data = {}
    for file_path in tqdm(file_list):
        logging.info(f"Retrived file to process: {file_path}")
        data = post_processing_batch_results(file_path)
        all_batch_data.update(data)

    logging.info("Number of data points: {}".format(len(all_batch_data)))
    all_original_data = {}
    with open(original_file, "r") as f:
        non_photograph_image_category_count = 0
        for line in f:
            data = json.loads(line)
            image_id = os.path.basename(data["image_path"])
            image_id = os.path.splitext(image_id)[0]
            if image_id in all_batch_data:
                data["QA_meta"]["open-ended"][0][
                    "en_question_hallucination"
                ] = all_batch_data[image_id][
                    "response"
                ]  # ["question"]
                data["model"] = all_batch_data[image_id]["model"]
                all_original_data[image_id] = data
            else:
                print(f"Image ID: {image_id} not found in batch data")
                # sys.exit()

    logging.info(
        "Number of data points with original: {}".format(len(all_original_data))
    )
    logging.info(
        f"Number of non-photograph image categories: {non_photograph_image_category_count}"
    )
    with open(output_file, "w") as f:
        for image_id, data in all_original_data.items():
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    logging.info(f"Output saved to: {output_file}")


def process_retrieved_data(batch_file, save_dir, original_file, output_file):
    file_list = []
    with open(batch_file, "r") as f:
        entries = f.readlines()
        for entry in entries:
            batch_id, file_path = entry.split(",")
            output_file_path = os.path.join(save_dir, f"batch_output_{batch_id}.jsonl")
            file_list.append(output_file_path.strip())

    all_batch_data = {}
    for file_path in tqdm(file_list):
        logging.info(f"Retrieved file to process: {file_path}")
        data = post_processing_batch_results(file_path)
        all_batch_data.update(data)

    logging.info(
        "Number of batch data points (retrived): {}".format(len(all_batch_data))
    )
    all_original_data = {}
    with open(original_file, "r") as f:
        non_photograph_image_category_count = 0
        for line in f:
            data = json.loads(line)
            image_id = os.path.basename(data["image_path"])
            image_id = os.path.splitext(image_id)[0]
            if image_id in all_batch_data:
                data["QA_meta"]["open-ended"][0][
                    "en_question_hallucination"
                ] = all_batch_data[image_id]["response"]["question"]
                data["model"] = all_batch_data[image_id]["model"]
                all_original_data[image_id] = data
            else:
                print(f"Image ID: {image_id} not found in batch data...")
                # sys.exit()

    logging.info(
        "Number of data points with original (retrived): {}".format(
            len(all_original_data)
        )
    )
    logging.info(
        f"Number of non-photograph image categories: {non_photograph_image_category_count}"
    )
    with open(output_file, "w") as f:
        for image_id, data in all_original_data.items():
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    logging.info(f"Output saved to: {output_file}")


def process_retrieved_error_data(batch_file, save_dir, original_file, output_file):
    file_list = []
    with open(batch_file, "r") as f:
        entries = f.readlines()
        for entry in entries:
            batch_id, file_path = entry.split(",")
            output_file_path = os.path.join(
                save_dir, f"batch_output_{batch_id}_error.jsonl"
            )
            file_list.append(output_file_path.strip())

    all_batch_data = []
    for file_path in tqdm(file_list):
        logging.info(f"Retrieved file to process: {file_path}")
        if not os.path.exists(file_path):
            continue
        with open(file_path, "r") as f:
            file_content = f.readlines()
        for line in file_content:
            try:
                json_object = json.loads(line)
                image_id = os.path.splitext(os.path.basename(json_object["custom_id"]))[
                    0
                ]
                all_batch_data.append(image_id)
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON in file {file_path}: {e}")
                continue

    logging.info(
        "Number of batch data points with error: {}".format(len(all_batch_data))
    )

    all_original_data = {}
    with open(original_file, "r") as f:
        for line in f:
            non_photograph_image_category_count = 0
            data = json.loads(line)
            image_id = os.path.basename(data["image_path"])
            image_id = os.path.splitext(image_id)[0]
            if image_id in all_batch_data:
                all_original_data[image_id] = data

    logging.info(
        "Number of data points with error - original: {}".format(len(all_original_data))
    )
    logging.info(
        f"Number of non-photograph image categories: {non_photograph_image_category_count}"
    )
    with open(output_file, "w") as f:
        for image_id, data in all_original_data.items():
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    logging.info(f"Output saved to: {output_file}")


def process_data(file_list, output_file):
    all_data = []
    for file_path in tqdm(file_list):
        print(file_path)
        data = post_processing_batch_results(file_path)
        all_data.extend(data)
    logging.info("Number of data points: {}".format(len(all_data)))
    with open(output_file, "w") as f:
        for data in all_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    logging.info(f"Output saved to: {output_file}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate QA batch retieval.")
    parser.add_argument(
        "--batch_file",
        type=str,
        default="cached_dir/QA/Qatar/batch_files.txt/",
        help="File to store batch file names",
    )
    parser.add_argument(
        "--original_file",
        type=str,
        default="cached_dir/QA/Qatar/data.jsonl",
        help="File to store batch file names",
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
        "--output_file",
        type=str,
        default="cached_dir/QA/Qatar/batch_files.jsonl/",
        help="Output file",
    )
    parser.add_argument(
        "--output_error_file",
        type=str,
        default="cached_dir/batch_files.jsonl",
        help="Output file",
    )
    parser.add_argument(
        "--retrieve",
        type=str,
        default="False",
        help="True/False to retrieve the batch files",
    )

    args = parser.parse_args()

    # Configure logging
    configure_logging()

    # Convert the string to a boolean
    args.retrieve = args.retrieve.lower() == "true"

    # Resolve paths
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if args.retrieve:
        env_path = args.env_file
        if not os.path.exists(env_path):
            logging.error(f"Error: {env_path} not found!")
            return

        # Load environment variables
        logging.info("Loading environment variables...")
        azure_openai_api_key, azure_openai_api_url, openai_api_version, _ = load_env(
            env_path
        )
        intermediate_batch_file_name = args.batch_file

        batch_manager = AzureOpenAIBatchManager(
            azure_openai_api_key,
            azure_openai_api_url,
            openai_api_version,
            batch_file_name=intermediate_batch_file_name,
        )

        file_list, error_file_list = batch_manager.retrieve_all_submitted_batches(
            intermediate_batch_file_name, batch_output_dir=output_dir
        )
        logging.info(f"Number of files: {len(file_list)}")

        process_batch_data(file_list, args.original_file, args.output_file)
        process_batch_error_data(
            error_file_list, args.original_file, args.output_error_file
        )
    else:
        process_retrieved_data(
            args.batch_file, output_dir, args.original_file, args.output_file
        )
        process_retrieved_error_data(
            args.batch_file, output_dir, args.original_file, args.output_error_file
        )


if __name__ == "__main__":
    main()
