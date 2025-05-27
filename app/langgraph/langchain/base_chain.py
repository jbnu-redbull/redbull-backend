from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar, Generic, List, Callable
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableSerializable, RunnableConfig, RunnableLambda
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.memory import BaseMemory
from langchain.memory import ConversationBufferMemory # 기본 인메모리용
import time
import asyncio
import logging
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, AsyncRetrying, RetryError, RetryCallState
from pydantic import BaseModel

from .llm_factory import LLMFactory, CreatedLLMInfo
from .settings import LangChainSettings

# 새로운 파서 임포트
from .response.parser import ModelOutputParser
from .response.base_response import CustomResponseModel

# 로거 설정
logger = logging.getLogger(__name__)

class ChainError(Exception):
    """Base exception for chain-related errors."""
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.original_exception = original_exception

class ModelError(ChainError):
    """Base class for model-related errors."""

class ModelConnectionError(ModelError):
    """Raised when there's an issue connecting to the model provider."""

class ModelTimeoutError(ModelError):
    """Raised when a model request times out."""

class ModelRateLimitError(ModelError):
    """Raised when API rate limits are exceeded."""

# OutputParserException은 langchain_core.exceptions에서 직접 가져다 쓸 수 있음
from langchain_core.exceptions import OutputParserException

# --- Retry Configuration ---
# DEFAULT_RETRY_ATTEMPTS = 3
# DEFAULT_RETRY_WAIT_EXPONENTIAL_MULTIPLIER = 1
# DEFAULT_RETRY_WAIT_EXPONENTIAL_MIN = 1
# DEFAULT_RETRY_WAIT_EXPONENTIAL_MAX = 60

# 제네릭 타입을 위한 TypeVar 정의
ChainOutput = TypeVar('ChainOutput')

class BaseChain(ABC, Generic[ChainOutput]):
    """
    Abstract base class for creating language model chains with built-in retry,
    error handling, and provider abstraction.
    """
    llm: BaseChatModel
    parser: BaseOutputParser[ChainOutput] # 파서의 반환타입을 ChainOutput으로 명시
    prompt_template_str: str
    llm_chain: RunnableSerializable[Dict[str, Any], ChainOutput] # Runnable의 반환타입도 ChainOutput
    
    settings: LangChainSettings
    memory: BaseMemory # 내부 메모리 객체
    memory_input_key: str
    memory_history_key: str
    # memory_output_key는 save_context에서 사용되지만, ConversationBufferMemory의 기본값을 따를 수 있음

    effective_llm_provider: str
    effective_model_name: str
    effective_model_alias: Optional[str]

    # Retry settings from LangChainSettings.retry_settings
    retry_attempts: int
    retry_wait_multiplier: int
    retry_wait_min: int
    retry_wait_max: int
    retryable_exceptions: List[Type[Exception]]

    
    def __init__(
        self,
        langchain_settings_obj: LangChainSettings,
        prompt_template_str: str,
        output_response_model: Optional[Type[CustomResponseModel]] = None,
        provider_name_override: Optional[str] = None,
        model_alias_override: Optional[str] = None,
        model_name_override: Optional[str] = None, # 실제 LLM 모델명 직접 지정
        custom_retryable_exceptions: Optional[List[Type[Exception]]] = None,
        memory_instance: Optional[BaseMemory] = None,
        memory_input_key: str = "input", 
        memory_history_key: str = "history",
        **kwargs: Any # LLMFactory.create_llm으로 전달될 추가 LLM 파라미터
    ):
        if langchain_settings_obj is None:
            raise ValueError("langchain_settings_obj must be provided.")
        self.settings = langchain_settings_obj
        self.prompt_template_str = prompt_template_str
        self.memory_input_key = memory_input_key
        self.memory_history_key = memory_history_key
        
        logger.info(
            f"Initializing BaseChain. Provider Override: '{provider_name_override}', "
            f"Model Alias Override: '{model_alias_override}', Model Name Override: '{model_name_override}'. "
            f"Memory Input Key: '{memory_input_key}', Memory History Key: '{memory_history_key}'. "
            f"LLM kwargs: {kwargs}"
        )

        if memory_instance:
            self.memory = memory_instance
            logger.info(f"Using provided memory instance: {type(self.memory).__name__}")
        else:
            self.memory = ConversationBufferMemory(
                memory_key=self.memory_history_key, 
                input_key=self.memory_input_key,
                return_messages=False # LangChain 문자열 히스토리로 사용
            )
            logger.info(f"No memory_instance provided. Initialized default ConversationBufferMemory.")

        self.retry_attempts = self.settings.retry_settings.retry_max_attempts
        self.retry_wait_multiplier = self.settings.retry_settings.retry_wait_multiplier
        self.retry_wait_min = self.settings.retry_settings.retry_min_interval_seconds
        self.retry_wait_max = self.settings.retry_settings.retry_max_interval_seconds
        
        self.retryable_exceptions = [
            ModelConnectionError, ModelTimeoutError, ModelRateLimitError
        ]
        if custom_retryable_exceptions:
            self.retryable_exceptions.extend(custom_retryable_exceptions)

        if output_response_model:
            if not issubclass(output_response_model, BaseModel):
                raise ValueError("output_response_model must be a Pydantic BaseModel class (ideally CustomResponseModel).")
            self.parser = ModelOutputParser(pydantic_object=output_response_model)
            logger.info(f"Using ModelOutputParser with model: {output_response_model.__name__}")
        else:
            logger.warning("No output_response_model provided. Defaulting to StrOutputParser. Output will be a string.")
            self.parser = StrOutputParser()

        try:
            # LLMFactory.create_llm 호출하고 CreatedLLMInfo 객체를 받음
            created_llm_info: CreatedLLMInfo = LLMFactory.create_llm(
                settings=self.settings,
                provider_name_override=provider_name_override,
                model_alias_override=model_alias_override,
                model_name_override=model_name_override,
                **kwargs 
            )
            
            # 반환된 정보를 사용하여 필드 설정
            self.llm = created_llm_info.llm_instance
            self.effective_llm_provider = created_llm_info.provider_name
            self.effective_model_name = created_llm_info.model_name
            self.effective_model_alias = created_llm_info.model_alias

            logger.info(f"LLM for BaseChain initialized. Provider: '{self.effective_llm_provider}', Effective Model Name: '{self.effective_model_name}', Effective Alias: '{self.effective_model_alias}'")

        except ValueError as ve:
            logger.error(f"LLMFactory creation error: {ve}", exc_info=True)
            raise ModelError(f"Failed to create LLM via factory: {ve}", original_exception=ve)
        except Exception as e:
            logger.error(f"Unexpected error creating LLM model: {e}", exc_info=True)
            raise ModelError(f"Unexpected error creating LLM model: {e}", original_exception=e)

        self._rebuild_chain_with_memory()

    def get_memory(self) -> BaseMemory:
        """Returns the current memory instance."""
        return self.memory

    def set_memory(self, memory_instance: BaseMemory):
        """Sets or replaces the memory instance and rebuilds the chain."""
        if not isinstance(memory_instance, BaseMemory):
            raise TypeError("memory_instance must be a subclass of BaseMemory.")
        logger.info(f"Setting new memory instance: {type(memory_instance).__name__}")
        self.memory = memory_instance
        self._rebuild_chain_with_memory()

    def _save_memory_and_extract_output(self, chain_result_with_inputs: Dict[str, Any]) -> ChainOutput:
        """
        Helper function to save context to memory and extract the final LLM output.
        Assumes chain_result_with_inputs contains 'original_input' and 'llm_output'.
        """
        original_user_inputs = chain_result_with_inputs.get('original_input', {})
        llm_result = chain_result_with_inputs.get('llm_output')

        if not isinstance(original_user_inputs, dict):
            logger.error(f"Expected 'original_input' to be a dict, but got {type(original_user_inputs)}. Cannot save context accurately.")
            return llm_result # type: ignore

        mem_inputs_for_save = {
            key: value for key, value in original_user_inputs.items() 
            if key == self.memory_input_key 
        }
        if not mem_inputs_for_save and self.memory_input_key in original_user_inputs: 
             mem_inputs_for_save = {self.memory_input_key: original_user_inputs[self.memory_input_key]}


        if not mem_inputs_for_save:
             logger.warning(f"Could not determine specific input for memory from 'original_input' using key '{self.memory_input_key}'. "
                            f"Saving context might be incomplete. Original input keys: {list(original_user_inputs.keys())}")
        
        if isinstance(llm_result, str):
            mem_outputs_for_save = {"output": llm_result}
        elif hasattr(llm_result, 'model_dump_json'): 
            mem_outputs_for_save = {"output": str(llm_result)}
            logger.debug(f"Saving Pydantic model output to memory as string: {str(llm_result)[:100]}...")
        else:
            mem_outputs_for_save = {"output": str(llm_result)}
            logger.debug(f"Saving non-string, non-Pydantic output to memory as string: {str(llm_result)[:100]}...")

        try:
            self.memory.save_context(mem_inputs_for_save, mem_outputs_for_save)
            logger.debug(f"Saved context to memory. Inputs: {mem_inputs_for_save}, Outputs: {mem_outputs_for_save}")
        except Exception as e:
            logger.error(f"Error saving context to memory: {e}", exc_info=True)

        return llm_result # type: ignore

    def _rebuild_chain_with_memory(self):
        prompt = self._create_prompt(self.prompt_template_str, self.parser)

        def prepare_inputs_for_prompt(input_data: Dict[str, Any]) -> Dict[str, Any]:
            # Pass all input_data to load_memory_variables, as some memory types might use other keys.
            history = self.memory.load_memory_variables(input_data).get(self.memory_history_key, "")
            
            prompt_inputs = input_data.copy() 
            prompt_inputs[self.memory_history_key] = history 
            
            if self.memory_input_key not in prompt_inputs:
                prompt_inputs[self.memory_input_key] = "" 

            logger.debug(f"Inputs prepared for prompt: keys={list(prompt_inputs.keys())}, history_snippet='{str(history)[:100]}...'")
            return prompt_inputs
        
        core_chain_logic = RunnableLambda(prepare_inputs_for_prompt) | prompt | self.llm | self.parser

        self.llm_chain = RunnablePassthrough.assign(original_input=lambda x: x.copy()) | \
                         RunnablePassthrough.assign(llm_output=core_chain_logic) | \
                         RunnableLambda(self._save_memory_and_extract_output)
        
        logger.info("LLM chain rebuilt with integrated memory loading and saving.")

    def _create_prompt(self, template_str: str, parser: BaseOutputParser) -> PromptTemplate:
        try:
            temp_prompt = PromptTemplate.from_template(template_str)
            input_variables = temp_prompt.input_variables
            logger.debug(f"Extracted input variables using PromptTemplate.from_template: {input_variables}")
        except Exception as e:
            logger.warning(f"Could not reliably extract input_variables for template. Error: {e}. Falling back to regex.")
            import re
            input_variables = list(set(re.findall(r"\\{([^\\]+)\\}", template_str)))
            logger.debug(f"Extracted input variables using regex fallback: {input_variables}")

        if self.memory_history_key not in input_variables:
            logger.warning(
                f"Memory history key '{self.memory_history_key}' not found in prompt template's input_variables ({input_variables}). "
                f"Ensure template includes '{{{self.memory_history_key}}}'."
            )
        if self.memory_input_key not in input_variables:
             logger.warning(
                f"Memory input key '{self.memory_input_key}' not found in prompt template's input_variables ({input_variables}). "
                f"Ensure template includes '{{{self.memory_input_key}}}'."
            )

        partial_variables = {}
        format_instructions_key = "format_instructions"
        if format_instructions_key in input_variables:
            if hasattr(parser, 'get_format_instructions'):
                try:
                    format_instructions_content = parser.get_format_instructions()
                    partial_variables[format_instructions_key] = format_instructions_content
                    logger.info(f"Format instructions injected for parser: {type(parser).__name__}.")
                except Exception as e:
                    logger.warning(f"Could not get format instructions from parser {type(parser).__name__}: {e}.")
            else:
                 logger.warning(f"Parser {type(parser).__name__} lacks 'get_format_instructions'.")
        else:
            logger.debug(f"'{format_instructions_key}' not in template. No format instructions injected.")
            
        final_input_variables = [var for var in input_variables if var not in partial_variables]
        
        if self.memory_history_key not in final_input_variables:
            final_input_variables.append(self.memory_history_key)
        if self.memory_input_key not in final_input_variables:
            final_input_variables.append(self.memory_input_key)
        final_input_variables = sorted(list(set(final_input_variables)))


        return PromptTemplate(
            template=template_str,
            input_variables=final_input_variables,
            partial_variables=partial_variables
        )

    def _log_retry_attempt(self, retry_state: RetryCallState) -> None:
        logger.warning(
            f"Retrying LLM call (attempt {retry_state.attempt_number}) "
            f"for {self.effective_model_name} (Provider: {self.effective_llm_provider}) due to: {retry_state.outcome.exception()}. "
            f"Waiting {getattr(retry_state.next_action, 'sleep', 0):.2f} seconds."
        )

    def _classify_error(self, e: Exception) -> Exception:
        if isinstance(e, OutputParserException):
            return e 
        if isinstance(e, (ModelError, ChainError)):
            return e
        # TODO: Add more specific error classification based on provider/error messages
        return ChainError(f"An unexpected error occurred in the chain: {str(e)}", original_exception=e)

    def _invoke_with_retry(self, func: Callable[..., Any], *args, **kwargs) -> ChainOutput:
        tenacity_retry = retry(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(multiplier=self.retry_wait_multiplier, min=self.retry_wait_min, max=self.retry_wait_max),
            before_sleep=self._log_retry_attempt,
            retry=retry_if_exception_type(tuple(self.retryable_exceptions))
        )
        robust_call = tenacity_retry(func) # Apply retry decorator to the function directly
        try:
            start_time = time.monotonic()
            result = robust_call(*args, **kwargs) # Call the decorated function
            duration = time.monotonic() - start_time
            logger.info(f"LLM call to '{self.effective_model_name}' (Provider: '{self.effective_llm_provider}') successful in {duration:.2f}s.")
            return result
        except OutputParserException as ope:
            logger.error(f"Output parsing failed after LLM call: {ope}", exc_info=True)
            raise
        except RetryError as re:
            # This will be raised if all retry attempts fail for retryable exceptions
            original_err = re.last_attempt.exception()
            logger.error(f"LLM call to '{self.effective_model_name}' failed after {self.retry_attempts} attempts. Last error: {original_err}", exc_info=original_err)
            raise self._classify_error(original_err if original_err else re)
        except Exception as e:
            logger.error(f"LLM call failed with an unhandled exception: {e}", exc_info=True)
            raise self._classify_error(e)

    async def _ainvoke_with_retry(self, func: Callable[..., Any], *args, **kwargs) -> ChainOutput:
        tenacity_async_retry = AsyncRetrying(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(multiplier=self.retry_wait_multiplier, min=self.retry_wait_min, max=self.retry_wait_max),
            before_sleep=self._log_retry_attempt,
            retry=retry_if_exception_type(tuple(self.retryable_exceptions)),
            reraise=True
        )
        try:
            start_time = time.monotonic()
            result = await tenacity_async_retry(func, *args, **kwargs)
            duration = time.monotonic() - start_time
            logger.info(f"Async LLM call to '{self.effective_model_name}' (Provider: '{self.effective_llm_provider}') successful in {duration:.2f}s.")
            return result
        except OutputParserException as ope:
            logger.error(f"Async output parsing failed after LLM call: {ope}", exc_info=True)
            raise
        except RetryError as re: # Raised by tenacity if all retries fail
            original_err = re.last_attempt.exception()
            logger.error(f"Async LLM call to '{self.effective_model_name}' failed after {self.retry_attempts} attempts. Last error: {original_err}", exc_info=original_err)
            raise self._classify_error(original_err if original_err else re)
        except Exception as e:
            logger.error(f"Async LLM call failed with an unhandled exception: {e}", exc_info=True)
            raise self._classify_error(e)

    def invoke(self, input_data: Dict[str, Any], config: Optional[RunnableConfig] = None, **kwargs) -> ChainOutput:
        """Invoke the LLM chain with the given input data and retry logic."""
        logger.debug(f"Invoking chain for '{self.effective_model_name}' with input: {json.dumps(input_data, indent=2) if logger.getEffectiveLevel() == logging.DEBUG else list(input_data.keys())}")
        
        # kwargs can be passed to runnable.invoke, these are runtime parameters
        # BaseChain's __init__ kwargs are for LLM *creation*
        # Here, `kwargs` might be for `RunnableConfig` or other runtime aspects.
        run_config = config or RunnableConfig()
        if kwargs:
            run_config.update(kwargs)
            
        return self._invoke_with_retry(self.llm_chain.invoke, input_data, config=run_config)

    async def ainvoke(self, input_data: Dict[str, Any], config: Optional[RunnableConfig] = None, **kwargs) -> ChainOutput:
        """Asynchronously invoke the LLM chain with the given input data and retry logic."""
        logger.debug(f"Asynchronously invoking chain for '{self.effective_model_name}' with input keys: {list(input_data.keys())}")
        
        run_config = config or RunnableConfig()
        if kwargs:
            run_config.update(kwargs)

        return await self._ainvoke_with_retry(self.llm_chain.ainvoke, input_data, config=run_config)

    def stream(self, input_data: Dict[str, Any], config: Optional[RunnableConfig] = None, **kwargs) -> Any:
        """Stream the output of the LLM chain."""
        logger.debug(f"Streaming chain for '{self.effective_model_name}' with input keys: {list(input_data.keys())}")
        # Retry logic for streaming is more complex as it's an iterator.
        # Tenacity doesn't directly support retrying generators out-of-the-box in a simple way.
        # For now, streaming calls will not have the same automatic retry mechanism as invoke/ainvoke.
        # A custom retry wrapper for iterators would be needed.
        # Consider langchain's own RetryingDynamicRunnable if applicable.
        run_config = config or RunnableConfig()
        if kwargs:
            run_config.update(kwargs)
        try:
            return self.llm_chain.stream(input_data, config=run_config)
        except Exception as e:
            logger.error(f"LLM stream call failed: {e}", exc_info=True)
            raise self._classify_error(e)

    async def astream(self, input_data: Dict[str, Any], config: Optional[RunnableConfig] = None, **kwargs) -> Any:
        """Asynchronously stream the output of the LLM chain."""
        logger.debug(f"Asynchronously streaming chain for '{self.effective_model_name}' with input keys: {list(input_data.keys())}")
        run_config = config or RunnableConfig()
        if kwargs:
            run_config.update(kwargs)
        try:
            return self.llm_chain.astream(input_data, config=run_config)
        except Exception as e:
            logger.error(f"Async LLM stream call failed: {e}", exc_info=True)
            raise self._classify_error(e)

    @classmethod
    def from_template_and_model(
        cls,
        langchain_settings_obj: LangChainSettings,
        prompt_template_str: str,
        output_response_model: Type[CustomResponseModel],
        provider_name_override: Optional[str] = None,
        model_alias_override: Optional[str] = None,
        model_name_override: Optional[str] = None,
        memory_instance: Optional[BaseMemory] = None,
        memory_input_key: str = "input",
        memory_history_key: str = "history",
        **kwargs # Passed to __init__, then to LLMFactory.create_llm
    ) -> 'BaseChain[CustomResponseModel]': # Return type annotation
        logger.info(f"Creating BaseChain from template. Provider: '{provider_name_override or langchain_settings_obj.active_model_provider}', Alias: '{model_alias_override}', ModelName: '{model_name_override}', Memory: {type(memory_instance).__name__ if memory_instance else 'DefaultConversationBufferMemory'}")
        return cls(
            langchain_settings_obj=langchain_settings_obj,
            prompt_template_str=prompt_template_str,
            output_response_model=output_response_model,
            provider_name_override=provider_name_override,
            model_alias_override=model_alias_override,
            model_name_override=model_name_override,
            memory_instance=memory_instance,
            memory_input_key=memory_input_key,
            memory_history_key=memory_history_key,
            **kwargs
        )
