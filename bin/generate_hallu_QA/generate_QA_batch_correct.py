import base64
import json
import os

import openai
import pandas as pd
import requests
from openai import AzureOpenAI, OpenAI


# Provided configuration dictionaries
GPT_DICT = {
    "MODEL_NAME": "gpt-4.1-global-batch",
    "CHAT_REQUEST_URL": "/chat/completions",
    # "DESCRIPTION_PROMPT": "",
    # "USER_PROMPT": "",
    "MAX_TOKENS": 2000,
}

IMAGE_PARAM_DICT = {"APPLY_ENCODING": True, "SCALING": 0.5, "MINIMUM_SIZE": 1200}


def get_dictionary_keys(parsed_response, key_variations):
    """
    Extract and map actual keys from a parsed JSON response to predefined key variations.

    Args:
        parsed_response (dict): The dictionary obtained from parsing a JSON response.
        key_variations (dict): A dictionary where keys are the standard names and values are lists of possible variations.

    Returns:
        dict: A dictionary mapping standard keys to actual keys found in parsed_response.
    """
    result_keys = {key: None for key in key_variations.keys()}
    for key, variations in key_variations.items():
        for variation in variations:
            if variation in parsed_response:
                result_keys[key] = variation
                break
    return result_keys


########################################################
############# ImageBatchProcessor #######################
########################################################


class ImageBatchProcessor:
    def __init__(
        self,
        input_source,
        output_dir,
        deployment_name="gpt-4.1-global-batch",
        batch_file_size_limit=180 * 1024 * 1024,
        image_size_limit=10 * 1024 * 1024,
    ):
        """
        Initializes the batch processor.

        Args:
        - input_source (str): Path to the CSV file or directory containing images.
        - output_dir (str): Directory where batch files will be saved.
        - batch_file_size_limit (int): Maximum size of the batch file in bytes.
        - image_size_limit (int): Maximum size of an individual image in bytes.
        """
        self.input_source = input_source
        self.output_dir = output_dir
        self.batch_file_size_limit = batch_file_size_limit
        self.image_size_limit = image_size_limit
        GPT_DICT["MODEL_NAME"] = deployment_name

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def read_image_objects(self, input_file):
        data = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                json_object = json.loads(line)
                image_path = json_object.get("image_path", "")
                if "/alt/vllms/exp_multimodal" in image_path:
                    json_object["image_path"] = image_path.replace(
                        "/alt/vllms/exp_multimodal/",
                        "/alt/llm-benchmarking/exp_multimodal_cultural_alignment/",
                    )
                    if (
                        "Sudan" in json_object["image_path"]
                        or "Syria" in json_object["image_path"]
                        or "Saudi_Arabia" in json_object["image_path"]
                        or "Morocco" in json_object["image_path"]
                        or "Kuwait" in json_object["image_path"]
                        or "Jordan" in json_object["image_path"]
                        or "UAE" in json_object["image_path"]
                        or "Bahrain" in json_object["image_path"]
                        or "Libya" in json_object["image_path"]
                        or "Oman" in json_object["image_path"]
                        or "Iraq" in json_object["image_path"]
                        or "Algeria" in json_object["image_path"]
                        or "Oman" in json_object["image_path"]
                        or "Tunisia" in json_object["image_path"]
                    ):
                        json_object["image_path"] = json_object["image_path"].replace(
                            "/img/", "/"
                        )

                    if not os.path.exists(json_object["image_path"]):
                        print(
                            f"Image path {json_object['image_path']} does not exist. Skipping this image."
                        )
                        continue
                else:
                    if not os.path.exists(json_object["image_path"]):
                        print(
                            f"Image path {json_object['image_path']} does not exist. Skipping this image."
                        )
                        continue

                    # Skip images that are not suitable for VQA
                    # continue
                # if "status" in json_object and json_object["status"] == "suitable":
                data.append(json_object)
        return data

    def load_images(self):
        """Loads images from a CSV file or directory."""
        if os.path.isdir(self.input_source):
            return [
                os.path.join(self.input_source, f)
                for f in os.listdir(self.input_source)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        elif os.path.isfile(self.input_source) and self.input_source.endswith(".csv"):
            df = pd.read_csv(self.input_source)
            return df["image_path"].tolist()
        elif os.path.isfile(self.input_source) and self.input_source.endswith(".jsonl"):
            data = self.read_image_objects(self.input_source)
            return data
        else:
            raise ValueError(
                "Invalid input source. Must be a directory or a CSV file with 'image_path' column."
            )

    def encode_image_base64(self, file_path, scaling=1.0, minimum_size=None):
        """Encodes the image as a base64 string."""
        with open(file_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        # Scaling or resizing logic can be added here if necessary.
        # This implementation assumes scaling logic is outside scope of this base64 encoding.
        return encoded_image

    def get_image_size(self, file_path):
        """Returns the size of the image file in bytes."""
        return os.path.getsize(file_path)

    def generate_payload_message(
        self, image_path, question, answer, description, hallu_json
    ):
        """
        Generate the payload message for the OpenAI API request.

        Args:
            image_data (str): The image data to be included in the payload.

        Returns:
            list: The payload message formatted for the OpenAI API.
        """

        # system_prompt = """You fix a hallucinated question by replacing ONLY the marked substrings.

        # Rules
        # - Input provides {"question": "...", "spans": [...]} where all spans refer to the same hallucinated detail.
        # - Replace EVERY span with a concise, factual phrase supported by the Description.
        # - Keep the rest of the question EXACTLY unchanged (wording, spacing, casing, punctuation).
        # - Preserve the original question type (color/count/attribute/spatial/text/action).
        # - Single sentence, ≤ 18 words. No negations or meta language.

        # Output (STRICT JSON ONLY)
        # { "corrected_question": "<string>" }
        # """

        # system_prompt = """You fix a hallucinated question by replacing ONLY the substrings wrapped in [[...]].

        # Rules
        # - Input provides question_marked and spans (array of {"text": string}); all spans refer to the SAME hallucinated detail.
        # - Replace EVERY [[...]] with a concise, factual phrase supported by the Description.
        # - Keep the rest of the question EXACTLY unchanged (spacing, casing, punctuation).
        # - Preserve the original question type (color/count/attribute/spatial/text/action).
        # - Single sentence, ≤ 18 words. No negations or meta language.

        # Output (STRICT JSON ONLY)
        # {
        # "corrected_question": "<string>",
        # "spans": [
        #     { "text": "<exact text between [[...]]>", "replacement": "<factual phrase used>" },
        #     ...
        # ]
        # }

        # Constraints
        # - The number of items in spans MUST equal the number of [[...]] tags in question_marked, in reading order.
        # - Each spans[i].text MUST exactly match the i-th tagged substring (without the brackets).
        # - All replacements MUST be identical if they refer to the same entity/phrase (usual case).
        # - corrected_question MUST equal question_marked with each [[...]] replaced by the corresponding spans[i].replacement.
        # - Do not include any fields other than corrected_question and spans.
        # """

        system_prompt = f"""
        You fix a hallucinated question, then apply a minimal paraphrase.

        Process:
        1) Correct the question by replacing exactly the marked substring [[...]] with a truthful phrase grounded in Description.
        - Prefer using "orig_span" if it is correct.
        - Preserve all other tokens unchanged.
        2) Apply a MICRO-PARAPHRASE that keeps the SAME question type (e.g., name/color/count/where/text/action) and still targets the same answer.
        Allowed (choose ONE):
        - swap "What is the name of …" → "Which landmark is …" (still asks for a NAME),
        - reorder adjectives or add ONE grounded modifier already present in Description (e.g., location),
        - minor synonym/function-word swap.
        Disallowed: changing question type, adding new information, or altering the answer required.

        Constraints:
        - Single sentence, ≤ 18 words.
        - No negations or meta language.
        - The final question MUST be answerable by CANONICAL_ANSWER (exact name or obvious equivalent).

        You will be given a hallucinated JSON with this shape:

        {{
        "question_marked": "<the question with [[...]] around each hallucinated fragment>",                                
        "spans": [ {{"text": "<the exact hallucinated fragment between [[...]]"}}, ... ]
        "orig_span": string                                 
        }}

        Output STRICT JSON:
        {{ "corrected_question": string, "self_answer": string }}
        """

        user_prompt = f"""
        Description: {description}
        Original_Q: {question}
        Canonical answer: {answer}
        Hallucinated JSON:
        {json.dumps(hallu_json, ensure_ascii=False)}
        """

        if IMAGE_PARAM_DICT["APPLY_ENCODING"]:
            base64_image = self.encode_image_base64(
                image_path,
                scaling=IMAGE_PARAM_DICT["SCALING"],
                minimum_size=IMAGE_PARAM_DICT["MINIMUM_SIZE"],
            )
            image_path = f"data:image/jpeg;base64,{base64_image}"
        else:
            image_path = image_path

        payload_message = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                    # {"type": "image_url", "image_url": {"url": image_path}},
                ],
            },
        ]
        return payload_message

    def create_batches(self):
        """Creates batches of images based on size limits."""
        image_objects = self.load_images()
        current_batch = []
        current_batch_size = 0
        batch_counter = 1

        # Process images
        for img_obj in image_objects:
            file_path = img_obj["image_path"]
            description = img_obj["image_desc_meta"]["en_description"]
            hallu_question = img_obj["QA_meta"]["open-ended"][0][
                "en_question_hallucination"
            ]
            question = img_obj["QA_meta"]["open-ended"][0]["en_question"]
            answer = img_obj["QA_meta"]["open-ended"][0]["en_answer"]

            del hallu_question["question"]

            if self.get_image_size(file_path) <= self.image_size_limit:
                # Generate payload message using the new function
                payload_message = self.generate_payload_message(
                    file_path, question, answer, description, hallu_question
                )

                # Define task for batch processing based on chat_request_payload format
                task = {
                    "custom_id": os.path.basename(file_path),
                    "method": "POST",
                    "url": GPT_DICT["CHAT_REQUEST_URL"],
                    "body": {
                        "model": GPT_DICT["MODEL_NAME"],
                        "messages": payload_message,
                        "max_tokens": GPT_DICT["MAX_TOKENS"],
                        # "logprobs": True,
                        # "top_logprobs": 1,
                        "response_format": {"type": "json_object"},
                        "temperature": 0.5,
                    },
                }

                # Calculate task size
                task_size = len(json.dumps(task).encode("utf-8"))

                # Check if adding this task would exceed the batch size limit
                if current_batch_size + task_size > self.batch_file_size_limit:
                    # Save current batch
                    self.save_batch(current_batch, batch_counter)

                    # Reset for new batch
                    current_batch = []
                    current_batch_size = 0
                    batch_counter += 1

                # Add task to current batch
                current_batch.append(task)
                current_batch_size += task_size

        # Save any remaining tasks in the last batch
        if current_batch:
            self.save_batch(current_batch, batch_counter)

        print(f"Batch creation complete. {batch_counter} batch files created.")

    def save_batch(self, batch, batch_counter):
        """Saves the current batch to a JSONL file."""
        batch_file_path = os.path.join(self.output_dir, f"batch_{batch_counter}.jsonl")
        with open(batch_file_path, "w") as batch_file:
            for item in batch:
                batch_file.write(json.dumps(item) + "\n")
        print(f"Saved batch {batch_counter} to {batch_file_path}")


########################################################
############# OpenAIBatchManager #######################
########################################################


class AzureOpenAIBatchManager:
    def __init__(
        self,
        api_key,
        api_endpoint,
        api_version,
        deployment_name="gpt-4.1-global-batch",
        batch_file_name="submitted_batches.txt",
    ):
        """
        Initializes the batch manager with OpenAI API key and optional batch file name.

        Args:
            api_key (str): The OpenAI API key.
            batch_file_name (str): The file name for tracking submitted batch IDs.
        """
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_endpoint,
        )

        self.batch_file_name = batch_file_name
        GPT_DICT["MODEL_NAME"] = deployment_name

    def save_batch_id(self, batch_id, batch_file_path):
        """
        Save the batch ID and the corresponding JSON file name to a file.

        Args:
            batch_id (str): The ID of the batch.
            batch_file_path (str): The path of the JSON file submitted.
        """
        with open(self.batch_file_name, "a") as f:
            f.write(f"{batch_id},{batch_file_path}\n")
        print(f"Batch ID {batch_id} saved for file {batch_file_path}.")

    def submit_batch_job_from_file(self, batch_file_path, verbose=False):
        """
        Submit a single batch job from a JSON file to OpenAI and save the batch ID.

        Args:
            batch_file_path (str): Path to the JSON file containing batch requests.
            verbose (bool): If True, prints detailed information about submission.

        Returns:
            str: The ID of the submitted batch.
        """
        try:
            # Step 1: Create batch input file by uploading the JSONL file
            with open(batch_file_path, "rb") as file:
                batch_input_file = self.client.files.create(file=file, purpose="batch")

            batch_input_file_id = batch_input_file.id  # Get the uploaded file's ID

            # Step 2: Create the batch job using the uploaded file ID
            response = self.client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/chat/completions",  # Specify the correct endpoint
                completion_window="24h",  # Example completion window
                metadata={
                    "description": f"Batch job from {os.path.basename(batch_file_path)}"
                },
            )

            # Step 3: Extract and save the batch ID
            batch_id = response.id
            self.save_batch_id(
                batch_id, batch_file_path
            )  # Save batch ID and corresponding file path

            if verbose:
                print(f"Batch job submitted from {batch_file_path} with ID: {batch_id}")

            return batch_id

        except Exception as e:
            print(f"Error submitting batch job from {batch_file_path}: {e}")
            return None

    def submit_all_batches_in_directory(self, directory_path, verbose=False):
        """
        Submit all JSON files in a directory as separate batch jobs.

        Args:
            directory_path (str): The path to the directory containing JSON files.
        """
        submitted_file_list = []
        if os.path.isfile(self.batch_file_name):
            with open(self.batch_file_name, "r") as f:
                submitted_file_list = [
                    line.strip().split(",")[1] for line in f if "," in line
                ]

        for file_name in os.listdir(directory_path):
            if file_name.endswith(".jsonl"):
                if verbose:
                    print(f"Processing file {file_name}")
                file_path = os.path.join(directory_path, file_name)
                if file_path in submitted_file_list:
                    if verbose:
                        print(f"File {file_path} already submitted. Skipping.")
                    continue
                if verbose:
                    print(f"Submitting batch from {file_path}")
                self.submit_batch_job_from_file(file_path, verbose=verbose)

    def check_batch_status(self, batch_id):
        """
        Check the current status of a batch job using the batch ID.

        Args:
            batch_id (str): The ID of the batch job to check.

        Returns:
            str: The current status of the batch job.
        """
        try:
            response = self.client.batches.retrieve(batch_id)
            status = response.status
            print(f"Batch ID {batch_id} status: {status}")
            return status
        except Exception as e:
            print(f"Error checking status for batch {batch_id}: {e}")
            return "error"

    def fetch_batch_results(self, batch_id, save_dir=None):
        """
        Fetch the results of a completed batch job.

        Args:
            batch_id (str): The ID of the batch job to retrieve results for.

        Returns:
            list or None: The results of the batch job, or None if not completed.
        """
        try:
            try:
                response = self.client.batches.retrieve(batch_id)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    print(f"Batch {batch_id} not found.")
                    return None, None
                else:
                    raise e
            if response.status == "completed":
                response = self.client.batches.retrieve(batch_id)
                if response.output_file_id:
                    # Save the content to a JSONL file named after the batch_id
                    output_file_path = os.path.join(
                        save_dir, f"batch_output_{batch_id}.jsonl"
                    )
                    file_response = self.client.files.content(response.output_file_id)
                    with open(output_file_path, "w") as f:
                        f.write(file_response.text)

                    if response.error_file_id:
                        file_response = self.client.files.content(
                            response.error_file_id
                        )
                        output_file_path_error = os.path.join(
                            save_dir, f"batch_output_{batch_id}_error.jsonl"
                        )
                        with open(output_file_path_error, "w") as f:
                            f.write(file_response.text)
                    else:
                        output_file_path_error = None

                    print(f"Results retrieved for batch {batch_id}")
                    return output_file_path, output_file_path_error
                else:
                    print(f"No output file found for batch {batch_id}.")
                    return None, None
            else:
                print(f"Batch ID {batch_id} is not yet completed.")
                return None, None
        except Exception as e:
            print(f"Error fetching results for batch {batch_id}: {e}")
            return None, None

    def retrieve_all_submitted_batches(
        self, batch_file_name=None, batch_output_dir=None
    ):
        """
        Load all batch IDs from the specified file and retrieve their results.
        """
        file_list = []
        error_file_list = []
        try:
            if batch_file_name is None:
                current_batch_file_name = self.batch_file_name
            else:
                current_batch_file_name = batch_file_name

            with open(current_batch_file_name, "r") as f:
                batch_entries = f.read().splitlines()

            for entry in batch_entries:
                try:
                    batch_id, original_file = entry.split(",")
                    status = self.check_batch_status(batch_id)
                    if status == "completed":
                        (
                            output_file_path,
                            error_output_file_path,
                        ) = self.fetch_batch_results(
                            batch_id, save_dir=batch_output_dir
                        )
                        if output_file_path is None:
                            print(f"Failed to retrieve results for batch {batch_id}.")
                            continue
                        else:
                            file_list.append(output_file_path)

                        if error_output_file_path is None:
                            print(
                                f"Batch ID {batch_id} does not contain any error file."
                            )
                        else:
                            error_file_list.append(error_output_file_path)
                    else:
                        print(
                            f"Batch {batch_id} from file {original_file} is still in status: {status}"
                        )
                except Exception as ex:
                    print(f"Error processing batch entry {entry}: {ex}")
        except Exception as e:
            print(f"Error retrieving batch data: {e}")

        return file_list, error_file_list
