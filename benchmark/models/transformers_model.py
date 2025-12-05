import gc
import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, \
    BitsAndBytesConfig

from models.abstract_model import AbstractModel
from models.prompt_utils import create_prompt, generate_fws_exemplars


class TransformersModel(AbstractModel):
    def __init__(self, config, device_map, image_store_path):
        super().__init__()

        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)

        # Getting parameters from the configuration file
        model_name = config["model_path"]
        use_quantization = config.get("use_quantization", True)

        # Please note: there is a custom header for GPT-OSS
        # We do this in order to "freeze" the date in the prompt and
        # ensure reproducibility as much as possible  - see gpt-oss-20b-config.yaml for further information
        if "gpt-oss" in model_name:
            self.chat_header = (config.get("system_message", "") + config.get("reasoning_effort", "")
                                + config.get("chat_header", ""))
        else:
            self.chat_header = config.get("chat_header", "")

        self.chat_footer = config.get("chat_footer", "")
        self.fws_exemplars_path = config.get("fws_exemplars_path", "")
        self.tr_context_case = config.get("tr_context", "")
        self.device_map = device_map

        # Generation parameters
        self.batch_size = config.get("batch_size", 4)
        self.max_new_tokens = config.get("max_new_tokens", 64)

        # Initialize the model and tokenizer
        logger.info(f"Loading the tokenizer `{model_name}`...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            trust_remote_code=True,
            use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading the model `{model_name}`...")
        if use_quantization:
            # Quantization is set to 4B
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=False,
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map
            )

    def generate_predictions(self, inputs, no_context, cot, fws, mml):
        logger.info("Generating predictions...")

        exemplars = generate_fws_exemplars(self.fws_exemplars_path) if fws else ""

        prompts = inputs.apply(lambda row: self.chat_header +
                                           create_prompt(row, no_context, cot, exemplars, self.tr_context_case) +
                                           self.chat_footer, axis=1).tolist()

        outputs = []

        for i in tqdm(range(0, len(prompts), self.batch_size), desc="Generating predictions"):
            batch_prompts = prompts[i:i + self.batch_size]

            batch_inputs = self.tokenizer(
                batch_prompts,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt"
            )

            if isinstance(self.device_map, str):
                batch_inputs = {k: v.to(self.device_map) for k, v in batch_inputs.items()}

            with torch.inference_mode():
                output = self.llm.generate(
                    **batch_inputs,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,
                    return_dict_in_generate=False,
                    output_scores=False,
                )

            seq_cpu = output.detach().cpu()
            cut = batch_inputs["input_ids"].shape[1]
            gen_cpu = seq_cpu[:, cut:]
            model_outputs = self.tokenizer.batch_decode(gen_cpu, skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=True)

            outputs.extend(model_outputs)

            del output, seq_cpu, gen_cpu, batch_inputs

            # Cleanup of hanging/allocated space that may accumulate over long processing time
            # This is one every 15 batches - if done too often, may lead to memory fragmentation
            if (i // self.batch_size) % 15 == 0 and i > 0:
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated() / 2 ** 30
                reserved = torch.cuda.memory_reserved() / 2 ** 30
                if reserved - allocated > 6.0:
                    gc.collect()
                    torch.cuda.empty_cache()

        results = []
        logger.info("Saving results...")

        answers = inputs["answer_idx"].tolist()
        for inp, output in tqdm(zip(answers, outputs), total=len(answers), desc="Saving results"):
            clean_answer = self.extract_answer(output.strip())
            results.append({
                "true_label_idx": inp,
                "model_label": clean_answer,
                "model_answer": output.strip()
            })

        return results
