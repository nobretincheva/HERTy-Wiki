import json
import textwrap


def generate_base_prompt_information(inp, task_context):
    # while we are designing this benchmark for 4-MCQs we are adding more options
    # to avoid formatting errors
    option_labels = ["A", "B", "C", "D", "E", "F", "G"]
    answers = ""
    # to allow for reproducibility, answers are shuffled pre-emptively
    # minimal de-biasing strategy for MCQ
    for ans, option in zip(inp["answer_set"], option_labels):
        answers += f"""{option}. {ans} \n"""

    # this No Context setting is designed to establish if the LLM is purely relying on seeing
    # this typing in its pretraining (esp likely coming from Wikipedia)
    # no definitions for type should be present either
    if task_context:
        base_prompt = f"""This is about a Wikidata Entity. 

                    {inp["itemLabel"]} is a:
                    Options:  
                    {answers}  

                """
    else:
        base_prompt = f""" Below is information about a Wikidata Entity: the entity's name and short
                    description, as well as a number of statements about the entity in the format (Property: Value).

                    Entity: {inp["itemLabel"]}:
                    Description: {inp["itemDescription"]}

                    Statements:
                    {inp["properties"]}

                    Based on the information above, which of the following types (if any) is the most specific type that describes the entity?
                    Options:
                    {answers}

                """

    return base_prompt


def get_few_shot(few_shot_ex, cot, no_context):
    # NO domain specific exemplars for now
    # domain_specific = few_shot_ex[few_shot_ex["domain"] == domain]
    add_prompts = []
    for i, ex in enumerate(few_shot_ex):
        base = f"Example {i + 1}: \n" + generate_base_prompt_information(ex, no_context)
        base += """Please answer with the letter corresponding to the correct option, using this format: 
                    Answer: [<LETTER>]
            """
        if cot:
            base += ex["rationale"]

        base += "Answer: [" + ex["answer"] + "]"
        add_prompts.append(base)

    return "\n".join(add_prompts)


def set_up_image_tokens(inp, token, image_num):
    for i in range(image_num):
        # to allow for image tokens that have image id (which in a series is this image) -
        # have IMAGE_NUM in the token var:
        # e.g. phi-4-mul: <|image_IMAGE_NUM|> -> <|image_1|>
        if "IMAGENUM" in token:
            image_token = token.replace("IMAGENUM", f"{i}", 1)
        else:
            image_token = token

        inp = inp.replace("IMAGE_MISSING", image_token, 1)
    return inp


def create_prompt(inp, no_context, cot, exemplars, mml=False, image_token="<start_of_image>"):
    final_answer = """Please answer with the letter corresponding to the correct option, using this format: 
    Answer: [<LETTER>]
    """

    prompt = generate_base_prompt_information(inp, no_context)
    if mml:
        prompt = set_up_image_tokens(prompt, image_token, len(inp["image_ids"]))

    if exemplars:
        few_shot = get_few_shot(exemplars, cot, no_context)
        prompt = few_shot + "\n" + prompt
    elif cot:
        final_answer += """Let's think step by step: """

    # remove extra indentation
    return textwrap.dedent(prompt + final_answer)


def generate_fws_exemplars(fws_path):
    with open(fws_path) as f:
        exemplars = [json.loads(line) for line in f]

    return exemplars
