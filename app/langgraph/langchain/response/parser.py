from typing import Any, Dict, Union, Type, TypeVar

import json
import re
import asyncio
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, ValidationError

import logging
logger = logging.getLogger(__name__)

# CustomResponseModel 임포트
from .base_response import CustomResponseModel

# Pydantic 모델을 위한 제네릭 타입 변수 T의 바운드를 CustomResponseModel로 변경
T = TypeVar('T', bound=CustomResponseModel)

class ModelOutputParser(BaseOutputParser[T]):
    """
    Parses the LLM output into a Pydantic model instance that is expected
    to be a subclass of CustomResponseModel.
    Handles JSON extraction, parsing, and Pydantic model validation.
    Requires a Pydantic model class (subclass of CustomResponseModel) 
    to be provided during instantiation.
    """
    pydantic_object: Type[T]

    def __init__(self, pydantic_object: Type[T]):
        """Initialize with the Pydantic model class that the output should conform to."""
        super().__init__(pydantic_object=pydantic_object)
        # issubclass 체크는 CustomResponseModel로 변경하거나, BaseModel을 유지할 수 있습니다.
        # CustomResponseModel로 변경하면 더 엄격해집니다.
        if not issubclass(pydantic_object, CustomResponseModel):
            logger.error(f"Invalid pydantic_object type: {type(pydantic_object)}. Must be a subclass of CustomResponseModel.")
            raise TypeError("pydantic_object must be a subclass of CustomResponseModel (which inherits from pydantic.BaseModel)")
        logger.debug(f"ModelOutputParser initialized with CustomResponseModel: {pydantic_object.__name__}")

    def _extract_json_str(self, text: str) -> str:
        logger.debug(f"Attempting to extract JSON from text (first 100 chars): {text[:100]}...")
        cleaned_text = text.strip()
        
        match = re.search(r"^```(?:json)?\s*([\s\S]*?)\s*```$", cleaned_text, re.DOTALL | re.IGNORECASE)
        if match:
            extracted_str = match.group(1).strip()
            logger.debug(f"Extracted JSON using markdown code block regex. Length: {len(extracted_str)}")
            return extracted_str
        
        try:
            first_brace_index = cleaned_text.index('{')
            last_brace_index = cleaned_text.rindex('}')
            if last_brace_index > first_brace_index:
                extracted_str = cleaned_text[first_brace_index : last_brace_index + 1].strip()
                logger.debug(f"Extracted JSON using first '{{ ' and last '}}'. Length: {len(extracted_str)}")
                return extracted_str
        except ValueError:
            logger.debug("Could not find '{' and '}' in the expected order for JSON extraction.")
            pass 
        
        logger.debug("No specific JSON structure found (markdown, braces). Returning cleaned text as potential JSON.")
        return cleaned_text

    def parse(self, text: str) -> T:
        logger.info(f"Parsing LLM output for model {self.pydantic_object.__name__}. Input text length: {len(text)}")
        logger.debug(f"RAW LLM OUTPUT (for {self.pydantic_object.__name__}):\n{text}")
        json_str = self._extract_json_str(text)

        try:
            logger.debug(f"Attempting to parse JSON string (first 100 chars): {json_str[:100]}...")
            json_object = json.loads(json_str)
            logger.debug(f"Successfully parsed JSON string into Python dict.")
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError: {e}. Failed to parse extracted JSON string.", exc_info=True)
            logger.debug(f"Original text for failed parse (first 200): {text[:200]}...")
            logger.debug(f"Extracted JSON string for failed parse (first 200): {json_str[:200]}...")
            raise OutputParserException(
                f"Failed to parse LLM output as JSON. Error: {e}.\n"
                f"Original text:\n```\n{text}\n```\n"
                f"Attempted to parse (after JSON extraction):\n```\n{json_str}\n```",
                llm_output=text,
            )

        try:
            logger.debug(f"Attempting to validate parsed JSON with Pydantic model: {self.pydantic_object.__name__}")
            validated_model = self.pydantic_object.model_validate(json_object)
            logger.info(f"Successfully validated JSON against Pydantic model {self.pydantic_object.__name__}.")
            return validated_model
        except ValidationError as e:
            logger.error(f"Pydantic ValidationError for model {self.pydantic_object.__name__}: {e}", exc_info=True)
            error_detail = {
                "error_type": "PydanticValidationError",
                "model_name": self.pydantic_object.__name__,
                "errors": e.errors(),
                "input_data": json_object
            }
            raise OutputParserException(
                f"LLM output failed Pydantic model validation for {self.pydantic_object.__name__}.\n"
                f"Error details: {json.dumps(error_detail, indent=2, ensure_ascii=False)}",
                llm_output=text,
                observation=json.dumps(e.errors(), indent=2)
            )

    async def aparse(self, text: str) -> T:
        logger.info(f"Async parsing LLM output for model {self.pydantic_object.__name__}. Input text length: {len(text)}")
        logger.debug(f"ASYNC RAW LLM OUTPUT (for {self.pydantic_object.__name__}):\n{text}")
        json_str = self._extract_json_str(text) # _extract_json_str is synchronous, consider async if it becomes a bottleneck

        try:
            logger.debug(f"Async: Attempting to parse JSON string (first 100 chars): {json_str[:100]}...")
            json_object = await asyncio.to_thread(json.loads, json_str)
            logger.debug(f"Async: Successfully parsed JSON string into Python dict.")
        except json.JSONDecodeError as e:
            logger.error(f"Async JSONDecodeError: {e}. Failed to parse extracted JSON string.", exc_info=True)
            logger.debug(f"Async: Original text for failed parse (first 200): {text[:200]}...")
            logger.debug(f"Async: Extracted JSON string for failed parse (first 200): {json_str[:200]}...")
            raise OutputParserException(
                f"Failed to parse LLM output as JSON (async). Error: {e}.\n"
                f"Original text:\n```\n{text}\n```\n"
                f"Attempted to parse (after JSON extraction):\n```\n{json_str}\n```",
                llm_output=text,
            )

        try:
            logger.debug(f"Async: Attempting to validate parsed JSON with Pydantic model: {self.pydantic_object.__name__}")
            validated_model = await asyncio.to_thread(self.pydantic_object.model_validate, json_object)
            logger.info(f"Async: Successfully validated JSON against Pydantic model {self.pydantic_object.__name__}.")
            return validated_model
        except ValidationError as e:
            logger.error(f"Async Pydantic ValidationError for model {self.pydantic_object.__name__}: {e}", exc_info=True)
            error_detail = {
                "error_type": "PydanticValidationError",
                "model_name": self.pydantic_object.__name__,
                "errors": e.errors(),
                "input_data": json_object
            }
            raise OutputParserException(
                f"LLM output failed Pydantic model validation for {self.pydantic_object.__name__} (async).\n"
                f"Error details: {json.dumps(error_detail, indent=2, ensure_ascii=False)}",
                llm_output=text,
                observation=json.dumps(e.errors(), indent=2)
            )

    def get_format_instructions(self) -> str:
        logger.info(f"!!!! [ModelOutputParser] ENTERING get_format_instructions for {self.pydantic_object.__name__}")
        
        instructions = None # Ensure instructions is initialized

        if hasattr(self.pydantic_object, 'get_output_instructions') and \
           callable(getattr(self.pydantic_object, 'get_output_instructions')):
            logger.info(f"!!!! [ModelOutputParser] Found get_output_instructions on {self.pydantic_object.__name__}. Calling it.")
            try:
                instructions = self.pydantic_object.get_output_instructions()
                logger.info(f"!!!! [ModelOutputParser] Instructions from get_output_instructions: {'EMPTY' if not instructions else instructions[:200]}...") # Log more content
            except Exception as e:
                logger.error(f"!!!! [ModelOutputParser] ERROR calling get_output_instructions on {self.pydantic_object.__name__}: {e}", exc_info=True)
                instructions = "Error generating instructions. Check logs." # Fallback instruction on error
        else:
            logger.warning(
                f"!!!! [ModelOutputParser] {self.pydantic_object.__name__} does NOT have get_output_instructions or it's not callable. "
                f"Falling back to basic example generation."
            )
            example = self.pydantic_object.model_config.get("json_schema_extra", {}).get("example")
     
            if example is None:
                logger.warning(f"!!!! [ModelOutputParser] Fallback: No 'example' found in json_schema_extra for {self.pydantic_object.__name__}. Constructing a default example.")
                example_fields = {}
                for field_name, field_info in self.pydantic_object.model_fields.items():
                    if field_info.is_required():
                        example_fields[field_name] = f"example_{field_name}"
                    else:
                        pass 
                if not example_fields: 
                    example_fields = {"info": "Provide a JSON object based on the model schema"}
                example = example_fields

            instructions = (
                f"Your response MUST be a single valid JSON object ONLY. "
                f"Do NOT include any additional text, explanations, or markdown like ```json ... ```.\\n\\n"
                f"Return ONLY the JSON object that looks like the following example (fields and types must match the schema, values can differ based on your response, but the structure should be similar):\\n"
                f"{json.dumps(example, indent=2, ensure_ascii=False)}\\n\\n"
                f"The JSON must start with {{ and end with }}. No other text or formatting is allowed."
            )
            logger.info(f"!!!! [ModelOutputParser] Instructions from fallback: {instructions[:200]}...")
            
        if instructions is None: # Should not happen if logic is correct, but as a safeguard
            logger.error("!!!! [ModelOutputParser] CRITICAL: Instructions are None before returning. Defaulting to a generic instruction.")
            instructions = "Please provide your response in a valid JSON format."

        logger.info(f"!!!! [ModelOutputParser] FINAL instructions being returned (first 200 chars): {'EMPTY' if not instructions else instructions[:200]}...")
        return instructions
