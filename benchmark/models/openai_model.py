import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from loguru import logger
from openai import OpenAI

from models.image_store import ImageStore
from models.prompt_utils import create_prompt, generate_fws_exemplars
from models.abstract_model import AbstractModel


class OpenAIModel(AbstractModel):
    def __init__(self, config, device_map, image_store_path):
        super().__init__()
        self.model_name = config["model_path"]
        self.max_new_tokens = config.get("max_new_tokens", 64)
        self.fws_exemplars_path = config.get("fws_exemplars_path", "")
        self.tr_context_case = config.get("tr_context", "")
        self.api_key = config.get("api_key", "")
        self.batch_size = config.get("batch_size", 4)

        self.image_store = ImageStore(image_store_path)

    def generate_api_prompts(self, mml, prompt, image_ids):
        api_input = {"role": "user", "content": [
            {"type": "input_text", "text": prompt}
        ]}

        if mml:
            for image_id in image_ids:
                data_url = self.image_store.get_data_url(image_id)
                if not data_url:
                    continue
                api_input["content"].append({
                    "type": "input_image",
                    "image_url": data_url
                })

        return [api_input]

    def call_api(self, client, prompt, effort="medium"):
        try:
            response = client.responses.create(
                model=self.model_name,
                max_output_tokens=self.max_new_tokens,
                input=prompt,
                # Please note: only the latest OpenAI/Google models have a reasoning parameter;
                # comment the below out if using anything before GPT-5 / Gemini 2.5
                reasoning={"effort": effort},
            )

            model_output = response.output_text
            return {"answer": model_output}
        except Exception as e:
            logger.error(f"API error: {e}")
            return {"answer": "ERROR"}

    def generate_predictions(self, inputs, no_context, cot, fws, mml):
        logger.info("Generating predictions...")
        if self.api_key:
            client = OpenAI(api_key=self.api_key)
        else:
            env_key = os.getenv("OPENAI_API_KEY")
            if not env_key:
                logger.error("No API key provided ...")
                raise EnvironmentError("Missing OpenAI API key.")
            logger.info("Using API key from environment variable OPENAI_API_KEY.")
            client = OpenAI()

        exemplars = generate_fws_exemplars(self.fws_exemplars_path) if fws else ""

        prompts = inputs.apply(
            lambda row: self.generate_api_prompts(
                mml,
                create_prompt(row, no_context, cot, exemplars, self.tr_context_case),
                row["image_ids"],
            ),
            axis=1
        ).tolist()

        outputs = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = {
                executor.submit(
                    self.call_api, client, prompt, "low" if no_context else "medium"
                ): idx
                for idx, prompt in enumerate(prompts)
            }

            for future in tqdm(as_completed(futures), total=len(prompts), desc="Generating predictions"):
                idx = futures[future]
                outputs[idx] = future.result()

        results = []
        logger.info("Saving results...")

        answers = inputs["answer_idx"].tolist()
        for inp, output in tqdm(zip(answers, outputs), total=len(answers), desc="Saving results"):
            clean_answer = self.extract_answer(output["answer"].strip())
            results.append({
                "true_label": inp,
                "model_label": clean_answer,
                "model_answer": output["answer"].strip(),
            })

        return results
