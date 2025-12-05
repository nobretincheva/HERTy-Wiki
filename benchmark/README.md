# Running HERTy-Wiki

This document provides a minimal, reproducible workflow for running baseline models on the WikiReason benchmark.
All commands below have been tested with Python 3.9+.

### 1. Installation
Clone the repository and install dependencies:
```
git clone <repo-url> 
cd <repo-name>

pip install -r requirements.txt
```
Please ensure a working CUDA installation and  compatible versions of torch and transformers.

**Note**: Phi-4-multimodal requires a downgrade of the transformers library to transformers==4.48.2 

### 2. Dataset
The benchmark uses two primary files:

```
entity_typing.parquet            # Main benchmark file (text fields + metadata + image_ids)
multimodal/images.arrow          # Deduplicated multimodal image store (linked via image_ids)
```
Please note: Each row in entity_typing.parquet may reference zero or more image_ids.
To run the text-only evaluation you are NOT required to have downloaded the images.arrow file.

### 3. Configuration files
Model behavior is controlled through YAML configuration files located in configs/.

Each file specifies:
- **The model backend** (currently only Transformers and OpenAI models are supported)
- **Model name and generation parameters**
- Quantization - at present we support 4B quantization via BitsAndBytes. This is set in code via \[True/False] 
- \[Transformers only] **Model headers and footers**: to support a multitude of models and ensure reproducibility where possible we construct the chat templates ourselves. These can typically be found in a chat_template.json on the model's HF page.
- **for multimodal implementations**: an image_token to position where the image should be inserted
- **optional**: **path to few-shot examples** - please refer to exemplars/et_exemplars.jsonl for our few-shot setup.

We provide all configs used to run our experiments in the `configs` folder. Our experiments were ran on an A100 with 80GB of VRAM.

### 4. Running the benchmark

To run a zero-shot set-up on our benchmark use the following code:
`python run_benchmark.py -c configs/transformers.yaml -i PATH_TO_INPUT_FILE
`
- For our no-context baseline add the flag: `--no_context True`
- To enable Chain-of-Thought prompting: `--chain_of_thought True` OR `-cot True`
- To enable few-shot prompting: `--few_shot True` OR `-fws True`
- To run a multimodal evaluation: ` ---multimodal_file PATH_TO_IMAGE_STORE_FILE`
- To specify an output file: `-o PATH_TO_OUTPUT_FILE`

You can combine the mmlm, cot and fws tags together to run further experimental settings.

### 5. Output Format
Predictions are written in JSON Lines (.jsonl), one JSON object per input example.

#### Text-only runs

For the standard (text-only) benchmark, each line has the form:

```
{
  "true_label_idx": 3,
  "model_label": "human",
  "model_answer": "The correct type for this entity is: human."
}
```
where: 
- true_label_idx: the gold label index for this instance (as defined in the benchmark label schema).
- model_label: the parsed label chosen by the model (should be one of \[A/B/C/D/NaN]).
- model_answer: the raw textual answer returned by the model (after .strip()).

#### Multimodal runs

For multimodal models, sorting is applied internally to group samples in a batch with the same numbers of images. To keep a reference to the original row ordering, an additional field is included:

```
{
  "original_idx": 127,
  "true_label_idx": 3,
  "model_label": "human",
  "model_answer": "The correct type for this entity is: human."
}
```
where:
- original_idx: index of the example in the original input file before any internal sorting/batching.
- true_label_idx, model_label, model_answer: as above. 
 
This structure makes it easy to realign predictions with the original dataset rows and compare different models and configurations on the same benchmark split.

### 6. Evaluation
We provide an evaluation script to compute metrics for the Entity Typing benchmark, including:
- Hierarchical accuracy, precision, recall and macro-F1
- Sibling-case accuracy, precision, recall and macro-F1
- Accuracy, precision, recall and macro-F1 over depths
- Macro F1 per domain

A confusion matrix over depth predictions for the hierarchical case.

Please note, cases that the model fails to format the output correctly, are treated as incorrect.

To run the evaluation script, please use:

`python eval_entity_typing.py -i PATH_TO_INPUT_FILE -r PATH_TO_OUTPUT_FILE -t et
`

For multimodal runs, the command is the same: the evaluator automatically detects and uses the `original_idx` column (if present) to restore the original row order:
### 7. Running other models
To run a different model to the ones we have provided, you would need to create a dedicated config file.
Importantly, to run a Transformers model you would need the model's path and a set of headers and footers (if instruction-tuned / using a chat setting).
These can be found in chat_template.json in the model's folder. 

