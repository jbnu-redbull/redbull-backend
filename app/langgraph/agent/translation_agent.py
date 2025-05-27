from typing import Dict, Any, Optional, Type
from ..base_agent import BaseAgent
from .response.translation_model import TranslationResponse
from ..langchain.base_chain import ChainError, OutputParserException
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

DEFAULT_TRANSLATION_PROMPT_TEMPLATE = """
Translate the following text into {target_language}.
Original text: {text}

If you can detect the source language, please include it in your response.
Provide your response in a JSON format that strictly adheres to the following schema:
{format_instructions}
"""

class TranslationAgent(BaseAgent[TranslationResponse]):
    agent_type: str = "translation"
    response_model: Type[BaseModel] = TranslationResponse
    default_prompt_template: str = DEFAULT_TRANSLATION_PROMPT_TEMPLATE

    """
    An agent specialized in translating text using an LLM.
    It inherits from BaseAgent and uses TranslationResponse for structured output.
    """

    def translate(
        self, 
        text: str, 
        target_language: str, 
        source_language: Optional[str] = None
    ) -> TranslationResponse:
        """
        Translates the given text to the target language.

        Args:
            text: The text to translate.
            target_language: The language to translate the text into (e.g., "Korean", "English").
            source_language: Optional. The language of the original text. If not provided, the model may try to detect it.

        Returns:
            A TranslationResponse object containing the translation and other details.

        Raises:
            OutputParserException: If the model's output cannot be parsed into TranslationResponse.
            ChainError: For other chain-related errors (e.g., model connection, timeout).
        """
        input_data: Dict[str, Any] = {
            "text": text,
            "target_language": target_language
        }
        if source_language:
            # If source language is provided, we can adjust the prompt or simply pass it
            # For this template, it's implicitly handled by the LLM if mentioned in the input.
            # We can also modify the prompt to explicitly use it if desired.
            # For now, we'll just ensure 'text' includes it if we want the model to know.
            # Or, add a {source_language} variable to the prompt template.
            pass # Current prompt doesn't explicitly use a 'source_language' variable beyond its inclusion in 'text'

        logger.info(f"Attempting to translate text to {target_language} using {self.agent_type} agent. Input text: '{text[:50]}...' ")
        try:
            response = self.invoke(input_data)
            if not response.original_text:
                response.original_text = text 
            logger.info(f"Translation successful. Translated text: '{response.translated_text[:50]}...' Language: {response.target_language}")
            return response
        except OutputParserException as ope:
            logger.error(f"Failed to parse translation response for '{text[:50]}...': {ope}", exc_info=True)
            raise # Re-raise for the caller to handle
        except ChainError as ce:
            logger.error(f"Chain error during translation for '{text[:50]}...': {ce}", exc_info=True)
            raise # Re-raise
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"An unexpected error occurred during translation for '{text[:50]}...': {e}", exc_info=True)
            raise ChainError(f"An unexpected error occurred during translation: {str(e)}", original_exception=e)

    async def atranslate(
        self, 
        text: str, 
        target_language: str, 
        source_language: Optional[str] = None
    ) -> TranslationResponse:
        """
        Asynchronously translates the given text to the target language.

        Args:
            text: The text to translate.
            target_language: The language to translate the text into.
            source_language: Optional. The language of the original text.

        Returns:
            A TranslationResponse object.

        Raises:
            OutputParserException: If parsing fails.
            ChainError: For other chain errors.
        """
        input_data: Dict[str, Any] = {
            "text": text,
            "target_language": target_language
        }
        if source_language:
            pass

        logger.info(f"Attempting async translation to {target_language} using {self.agent_type} agent. Input text: '{text[:50]}...' ")
        try:
            response = await self.ainvoke(input_data)
            if not response.original_text:
                response.original_text = text
            logger.info(f"Async translation successful. Translated text: '{response.translated_text[:50]}...' Language: {response.target_language}")
            return response
        except OutputParserException as ope:
            logger.error(f"Failed to parse async translation response for '{text[:50]}...': {ope}", exc_info=True)
            raise
        except ChainError as ce:
            logger.error(f"Chain error during async translation for '{text[:50]}...': {ce}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during async translation for '{text[:50]}...': {e}", exc_info=True)
            raise ChainError(f"An unexpected error occurred during async translation: {str(e)}", original_exception=e)