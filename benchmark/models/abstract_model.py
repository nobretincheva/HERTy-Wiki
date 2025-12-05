from typing import List
import re
import pandas as pd


class AbstractModel:
    def __init__(self):
        super().__init__()

    def generate_predictions(self, inputs: pd.DataFrame, no_context: bool, cot: bool, fws: bool,
                             mml: bool) -> List[List[str]]:
        raise NotImplementedError

    @staticmethod
    def extract_answer(text):
        # we grab the last possible answer match
        matches = re.findall(r"Answer:\s*\[?([A-Z])]?", text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
        return "NONE"
