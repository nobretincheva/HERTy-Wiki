import gc
import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, \
    BitsAndBytesConfig, AutoProcessor
from transformers.image_processing_utils import BatchFeature

from models.image_store import ImageStore
from models.abstract_model import AbstractModel
from models.prompt_utils import create_prompt, generate_fws_exemplars


# This is a custom implementation of our multimodal class for Phi-4-multimodal
# This is necessary due to the present implementation of Phi-4-mml [as of 04/12/2025] which does not support batching
class PhiMultimodalTransformersModel(AbstractModel):
    def __init__(self, config, device_map, image_store_path):
        super().__init__()

        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)

        # Getting parameters from the configuration file
        model_name = config["model_path"]
        use_quantization = config.get("use_quantization", True)

        self.chat_header = config.get("chat_header", "")
        self.chat_footer = config.get("chat_footer", "")
        self.image_token = config.get("image_token", "")
        self.fws_exemplars_path = config.get("fws_exemplars_path", "")
        self.tr_context_case = config.get("tr_context", "")
        self.device_map = device_map

        # Generation parameters
        self.batch_size = config.get("batch_size", 4)
        self.max_new_tokens = config.get("max_new_tokens", 64)

        # Initialize the model, tokenizer and processor
        logger.info(f"Loading the processor `{model_name}`...")

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
            use_fast=True
        )
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        logger.info(f"Loading the model `{model_name}`...")
        if use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=False,
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map=device_map,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map=device_map,
                torch_dtype=torch.float16,
                _attn_implementation="flash_attention_2",
            )

        self.image_store = ImageStore(image_store_path)

    def pad_left(self, seqs: list[torch.Tensor], pad_token_id: int) -> torch.Tensor:

        max_len = max(len(seq) for seq in seqs)
        padded = torch.full((len(seqs), max_len), pad_token_id, dtype=seqs[0].dtype)
        for i, seq in enumerate(seqs):
            padded[i, -len(seq):] = seq
        return padded

    def stack_and_pad_phi_inputs(self,
                                 inputs_list: list[BatchFeature],
                                 pad_token_id: int,
                                 device: str = "cuda"
                                 ) -> BatchFeature:
        seqs = [bf["input_ids"][0] for bf in inputs_list]  # (L_i)
        new_input_ids = self.pad_left(seqs, pad_token_id=pad_token_id)
        attention_mask = (new_input_ids != pad_token_id).long()

        data = dict(
            input_ids=new_input_ids,
            attention_mask=attention_mask,
        )

        if "pixel_values" in inputs_list[0]:
            data["pixel_values"] = torch.cat(
                [bf["pixel_values"] for bf in inputs_list],
                dim=0,
            )
        if "image_sizes" in inputs_list[0]:
            data["image_sizes"] = torch.cat(
                [bf["image_sizes"] for bf in inputs_list],
                dim=0,
            )

        if "input_mode" in inputs_list[0]:
            data["input_mode"] = inputs_list[0]["input_mode"]
        else:
            data["input_mode"] = "multimodal" if "pixel_values" in data else "text"

        return BatchFeature(data).to(device)

    def generate_predictions(self, inputs, no_context, cot, fws, mml):
        logger.info("Generating predictions...")

        exemplars = generate_fws_exemplars(self.fws_exemplars_path) if fws else ""

        inputs["__orig_idx__"] = range(len(inputs))
        inputs["num_images"] = inputs["image_ids"].apply(len)
        inputs = inputs.sort_values("num_images", ascending=False).reset_index(drop=True)

        image_ids_all = inputs["image_ids"].tolist()
        prompts = inputs.apply(lambda row: self.chat_header +
                                           create_prompt(row, no_context, cot, exemplars, self.tr_context_case, mml,
                                                         self.image_token) +
                                           self.chat_footer, axis=1).tolist()

        outputs = []

        for i in tqdm(range(0, len(prompts), self.batch_size), desc="Generating predictions"):
            batch_prompts = prompts[i:i + self.batch_size]
            batch_ids = image_ids_all[i:i + self.batch_size]
            batch_images = [[self.image_store.get_image(iid) for iid in ids] for ids in batch_ids]

            batch_inputs = []

            for prompt, imgs in zip(batch_prompts, batch_images):

                if imgs:
                    single_inputs = self.processor(
                        text=prompt,
                        images=imgs,
                        return_tensors="pt",
                    )
                else:
                    single_inputs = self.processor(
                        text=prompt,
                        return_tensors="pt",
                    )

                batch_inputs.append(single_inputs)

            pad_id = self.processor.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.processor.tokenizer.eos_token_id

            # Stack to a single BatchFeature to allow for batching
            device = self.device_map if isinstance(self.device_map, str) else "cuda"
            batch_inputs = self.stack_and_pad_phi_inputs(
                batch_inputs,
                pad_token_id=pad_id,
                device=device,
            )

            with torch.inference_mode():
                output = self.llm.generate(
                    **batch_inputs,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    do_sample=False,
                    return_dict_in_generate=False,
                    output_scores=False,
                )

            seq_cpu = output.detach().cpu()
            cut = batch_inputs["input_ids"].shape[1]
            gen_cpu = seq_cpu[:, cut:]
            model_outputs = self.processor.tokenizer.batch_decode(gen_cpu, skip_special_tokens=True,
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
        og_idx = inputs["__orig_idx__"].tolist()
        for inp, og, output in tqdm(zip(answers, og_idx, outputs), total=len(answers), desc="Saving results"):
            clean_answer = self.extract_answer(output.strip())
            results.append({
                "original_idx": og,
                "true_label_idx": inp,
                "model_label": clean_answer,
                "model_answer": output.strip()
            })

        return results
