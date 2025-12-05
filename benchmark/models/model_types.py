from enum import Enum

from models.openai_model import OpenAIModel
from models.transformers_model import TransformersModel
from models.mmlm_transofrmers_model import MultimodalTransformersModel
from models.phi_mmlm_transformers_model import PhiMultimodalTransformersModel


class Models(Enum):
    TRANSFORMERS_MODEL = "transformers_model"
    OPENAI_MODEL = "openai_model"

    @staticmethod
    def get_model(model_name: str, mmlm=False):
        model = Models(model_name)
        if model == Models.TRANSFORMERS_MODEL and mmlm:
            if "phi" in model_name:
                return PhiMultimodalTransformersModel
            else:
                return MultimodalTransformersModel
        elif model == Models.TRANSFORMERS_MODEL:
            return TransformersModel
        elif model == Models.OPENAI_MODEL:
            return OpenAIModel
        else:
            raise ValueError(f"Model `{model_name}` not found.")