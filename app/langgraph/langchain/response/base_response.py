from pydantic import BaseModel
import json
import logging
from abc import ABC, abstractmethod
from pydantic_core import PydanticUndefined
from typing import Any, get_origin, get_args, Union

logger = logging.getLogger(__name__)

class CustomResponseModel(BaseModel, ABC):
    """
    A base Pydantic model for LLM responses that includes a method 
    to generate custom output format instructions for the LLM.
    Requires subclasses to implement methods for getting a JSON example structure 
    and for formatting the final LLM instructions.
    """

    @staticmethod
    def _get_type_placeholder(field_type: Any, field_name: str = "unknown") -> Any:
        """
        Generates a simple placeholder value based on the field's type.
        Handles basic types, lists, dicts, and nested Pydantic models.
        This can be used by subclasses when implementing get_json_example_structure.
        """
        origin = get_origin(field_type)
        args = get_args(field_type)

        if origin is list or field_type is list:
            if args:
                return [CustomResponseModel._get_type_placeholder(args[0], f"{field_name}_item")]
            return []
        elif origin is dict or field_type is dict:
            if args and len(args) == 2:
                key_placeholder = CustomResponseModel._get_type_placeholder(args[0], f"{field_name}_key")
                if not isinstance(key_placeholder, (str, int, float, bool)):
                    key_placeholder = str(key_placeholder)
                return {key_placeholder: CustomResponseModel._get_type_placeholder(args[1], f"{field_name}_value")}
            return {}
        elif origin is Union:
            if args:
                for arg in args:
                    if arg is not type(None):
                        return CustomResponseModel._get_type_placeholder(arg, field_name)
            return None
        elif isinstance(field_type, type) and issubclass(field_type, BaseModel) and hasattr(field_type, 'get_json_example_structure'):
            # If it's a BaseModel subclass AND has get_json_example_structure (to avoid issues with BaseModels not from this hierarchy)
            try:
                return field_type.get_json_example_structure()
            except Exception as e:
                logger.error(f"Error getting example structure for nested model {field_type.__name__} in field {field_name}: {e}. Falling back to placeholder.")
                return { "_nested_model_error": f"Could not generate structure for {field_type.__name__}" }

        elif field_type is str:
            return "string"
        elif field_type is int:
            return 0
        elif field_type is float:
            return 0.0
        elif field_type is bool:
            return False
        elif field_type is Any:
            return "any_value"
        elif field_type is type(None):
            return None
        else:
            logger.debug(f"Returning placeholder for unhandled type {field_type} for field {field_name}")
            return f"placeholder_for_{str(field_type).replace('~', '')}"

    @classmethod
    @abstractmethod
    def get_json_example_structure(cls) -> dict:
        """
        Abstract method that subclasses must implement.
        Should generate a dictionary representing the JSON structure of this Pydantic model,
        using appropriate example values for its fields.
        """
        pass

    @classmethod
    @abstractmethod
    def format_llm_instructions(cls, json_structure_example: dict) -> str:
        """
        Abstract method that subclasses must implement.
        Takes a JSON structure example dictionary and returns the final formatted 
        instruction string for the LLM.
        """
        pass

    @classmethod
    def get_output_instructions(cls) -> str:
        """
        Main method called by the parser to get format instructions.
        It generates a JSON example structure and then calls the subclass-defined 
        formatting method.
        """
        logger.debug(f"[{cls.__name__}] Getting JSON example structure for LLM instructions.")
        json_example = cls.get_json_example_structure()
        logger.debug(f"[{cls.__name__}] JSON example structure generated: {json.dumps(json_example, indent=2, ensure_ascii=False)[:300]}...")
        logger.debug(f"[{cls.__name__}] Calling format_llm_instructions with the structure.")
        return cls.format_llm_instructions(json_example)
