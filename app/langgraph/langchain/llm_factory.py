import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Type, Any
from langchain_core.language_models import BaseChatModel
import inspect # Added for checking abstract classes
from .settings import LangChainSettings, PROVIDER_SETTINGS_MAP, BaseProviderLLMSettings
from dataclasses import dataclass # 추가

@dataclass
class CreatedLLMInfo:
    llm_instance: BaseChatModel
    provider_name: str
    model_name: str    # 실제 사용된 LLM 모델명
    model_alias: Optional[str] # 사용된 YAML 내 모델 별칭

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def create_model(self, model_name_to_use: str, provider_settings: BaseProviderLLMSettings, llm_specific_kwargs: Dict[str, Any]) -> BaseChatModel:
        """Create and return a language model instance using resolved settings."""
        pass

    @classmethod
    def get_provider_name(cls) -> str:
        """Generates a standardized provider name from the class name."""
        name = cls.__name__.replace("Provider", "").lower()
        if not name: # Should not happen if class names are like "OpenAIProvider"
            raise ValueError(f"Could not determine provider name for {cls.__name__}")
        return name

class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation"""
    
    def create_model(self, model_name_to_use: str, provider_settings: BaseProviderLLMSettings, llm_specific_kwargs: Dict[str, Any]) -> BaseChatModel:
        from langchain_openai import ChatOpenAI
        
        model_params = {
            "model": model_name_to_use, # ChatOpenAI uses 'model'
            "api_key": provider_settings.model_api_key if hasattr(provider_settings, 'model_api_key') else None,
            "base_url": provider_settings.model_provider_url if hasattr(provider_settings, 'model_provider_url') else None,
            "temperature": provider_settings.model_temperature,
            "max_tokens": provider_settings.model_max_tokens,
            "timeout": provider_settings.model_timeout,
        }
        # Add any other OpenAI specific fields from provider_settings
        # e.g., organization, etc. Make sure they are part of OpenAISettings model
        if hasattr(provider_settings, 'organization_id') and provider_settings.organization_id: # Example
             model_params['organization'] = provider_settings.organization_id

        model_params.update(llm_specific_kwargs) # kwargs override/add parameters
        model_params = {k: v for k, v in model_params.items() if v is not None} # Clean None values
            
        logger.debug(f"Creating OpenAI model \'{model_name_to_use}\' with params: {model_params}")
        return ChatOpenAI(**model_params)

class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider implementation"""

    def create_model(self, model_name_to_use: str, provider_settings: BaseProviderLLMSettings, llm_specific_kwargs: Dict[str, Any]) -> BaseChatModel:
        from langchain_anthropic import ChatAnthropic

        model_params = {
            "model": model_name_to_use, # ChatAnthropic uses 'model'
            "anthropic_api_key": provider_settings.model_api_key if hasattr(provider_settings, 'model_api_key') else None,
            "base_url": provider_settings.model_provider_url if hasattr(provider_settings, 'model_provider_url') else None,
            "temperature": provider_settings.model_temperature,
            "max_tokens": provider_settings.model_max_tokens,
            "timeout": provider_settings.model_timeout,
        }
        model_params.update(llm_specific_kwargs)
        model_params = {k: v for k, v in model_params.items() if v is not None}

        if "model" not in model_params or model_params["model"] is None:
            raise ValueError("Model name ('model') must be specified for Anthropic and cannot be None.")

        logger.debug(f"Creating Anthropic model \'{model_name_to_use}\' with params: {model_params}")
        return ChatAnthropic(**model_params)

class OllamaProvider(LLMProvider):
    """Ollama LLM provider implementation for local models"""
    
    def create_model(self, model_name_to_use: str, provider_settings: BaseProviderLLMSettings, llm_specific_kwargs: Dict[str, Any]) -> BaseChatModel:
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "Could not import langchain_ollama. Please install it with `pip install langchain-ollama`"
            )

        model_params = {
            "model": model_name_to_use,
            "base_url": provider_settings.model_provider_url if hasattr(provider_settings, 'model_provider_url') and provider_settings.model_provider_url else "http://localhost:11434",
            "temperature": provider_settings.model_temperature,
            "timeout": provider_settings.model_timeout,
        }
        # Add Ollama specific fields from provider_settings (e.g., num_ctx, num_gpu from OllamaSettings)
        for field_name in provider_settings.model_fields.keys():
            if field_name not in ["model_alias", "model_name", "model_temperature", "model_max_tokens", "model_timeout", "model_provider_url", "model_api_key", "model_provider"] and hasattr(provider_settings, field_name):
                param_value = getattr(provider_settings, field_name)
                if param_value is not None:
                    model_params[field_name] = param_value
        
        model_params.update(llm_specific_kwargs)
        model_params = {k: v for k, v in model_params.items() if v is not None}
        
        if model_params.get("base_url") == "http://localhost:11434":
            logger.info("Ollama provider is using default base URL: http://localhost:11434")
        
        if "model" not in model_params or model_params["model"] is None:
             raise ValueError("Model name ('model') must be specified for Ollama and cannot be None.")
        
        logger.debug(f"Creating Ollama model \'{model_name_to_use}\' with params: {model_params}")
        try:
            return ChatOllama(**model_params)
        except Exception as e:
            logger.error(f"Failed to create Ollama model with params {model_params}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to create Ollama model '{model_params.get('model')}': {str(e)}")

class LLMFactory:
    """Factory for creating LLM providers.
    Automatically discovers and registers subclasses of LLMProvider.
    """
    _provider_classes: Dict[str, Type[LLMProvider]] = {}
    _initialized_provider_instances: Dict[str, LLMProvider] = {}
    _discovery_completed: bool = False

    @classmethod
    def _discover_provider_classes(cls):
        if cls._discovery_completed:
            return
        logger.info("Discovering LLM provider classes...")
        for provider_class in LLMProvider.__subclasses__():
            if not inspect.isabstract(provider_class):
                try:
                    provider_name = provider_class.get_provider_name()
                    if provider_name not in cls._provider_classes:
                        cls.register_provider_class(provider_name, provider_class)
                    else: # Should not happen if class names are unique
                        logger.warning(f"Provider class for {provider_name} already registered. Skipping {provider_class.__name__}.")
                except ValueError as e:
                     logger.error(f"Could not register provider class {provider_class.__name__}: {e}")
        cls._discovery_completed = True
    
    @classmethod
    def register_provider_class(cls, name: str, provider_class: Type[LLMProvider]):
        if not issubclass(provider_class, LLMProvider):
            raise TypeError(f"{provider_class.__name__} must be a subclass of LLMProvider")
        if inspect.isabstract(provider_class):
            raise TypeError(f"Cannot register abstract class {provider_class.__name__}")
        if name in cls._provider_classes:
            logger.warning(f"Provider class for {name} is already registered. Overwriting with {provider_class.__name__}.")
        cls._provider_classes[name] = provider_class
        logger.info(f"Registered provider class: {name} -> {provider_class.__name__}")

    @classmethod
    def get_provider_instance(cls, provider_name: str) -> LLMProvider:
        cls._discover_provider_classes()
        if provider_name not in cls._initialized_provider_instances:
            provider_class_to_instantiate = cls._provider_classes.get(provider_name)
            if not provider_class_to_instantiate:
                supported = list(cls._provider_classes.keys())
                raise ValueError(f"Provider class for '{provider_name}' not found. Supported: {supported}")
            try:
                cls._initialized_provider_instances[provider_name] = provider_class_to_instantiate()
            except Exception as e:
                logger.error(f"Error instantiating provider class {provider_class_to_instantiate.__name__}: {e}", exc_info=True)
                raise RuntimeError(f"Could not instantiate provider {provider_class_to_instantiate.__name__}: {e}")
        return cls._initialized_provider_instances[provider_name]

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Returns a list of available provider names."""
        cls._discover_provider_classes()
        return list(cls._provider_classes.keys())

    @staticmethod
    def create_llm(
        settings: LangChainSettings, 
        provider_name_override: Optional[str] = None, 
        model_alias_override: Optional[str] = None,
        model_name_override: Optional[str] = None, 
        **kwargs: Any # Additional LLM parameters to override or add
    ) -> CreatedLLMInfo: # 반환 타입 변경
        
        if settings is None:
            raise ValueError("LangChainSettings object must be provided.")

        # 1. Determine the target provider name
        target_provider_name = provider_name_override if provider_name_override is not None else settings.active_model_provider
        if not target_provider_name:
            raise ValueError("LLM provider name could not be determined. Set active_model_provider in settings or pass provider_name_override.")
        
        target_provider_name = target_provider_name.lower()
        if target_provider_name not in PROVIDER_SETTINGS_MAP:
             raise ValueError(f"Provider '{target_provider_name}' is not a known provider in PROVIDER_SETTINGS_MAP. Available: {list(PROVIDER_SETTINGS_MAP.keys())}")

        # 2. Get the specific provider settings object (e.g., settings.openai)
        # This object is already initialized with a specific model alias/name based on YAML/kwargs during LangChainSettings.__init__
        provider_settings_obj = getattr(settings, target_provider_name, None)

        # 3. Handle overrides for model_alias and potentially re-initialize provider_settings_obj
        # If model_alias_override is given and it's different from what provider_settings_obj is configured for,
        # we need to create a new, temporary provider settings instance for that specific alias.
        if model_alias_override is not None and (provider_settings_obj is None or provider_settings_obj.model_alias != model_alias_override):
            logger.info(f"Model alias override ('{model_alias_override}') differs from current settings for '{target_provider_name}' (or settings not found). Creating/re-fetching settings for this specific alias.")
            
            ProviderSpecificSettingsClass = PROVIDER_SETTINGS_MAP.get(target_provider_name)
            if not ProviderSpecificSettingsClass: # Should have been caught above
                raise ValueError(f"Cannot find settings class for provider '{target_provider_name}'.")

            # Get raw YAML configurations for this provider
            provider_yaml_configs_raw = settings.provider_configurations.get(target_provider_name, {})
            provider_models_yaml_raw = provider_yaml_configs_raw.get("models", {})
            provider_default_alias_yaml = provider_yaml_configs_raw.get("default_model_name")
            # Global alias from llm_settings.model_name (if any)
            global_alias_from_yaml = settings.provider_configurations.get("model_name") # This is incorrect, global alias is not in provider_configurations root

            # Correct way to get global model_alias from a hypothetical merged_yaml_llm_settings in settings
            # For now, assume it's not easily accessible here or not critical if provider default exists.
            # It's best if BaseProviderLLMSettings handles its own alias resolution logic primarily.
            # LangChainSettings.__init__ passes global_model_alias_from_yaml to each provider setting init.
            # Here, we simulate parts of that if we re-initialize.

            # Kwargs for the new provider settings instance. model_name here is the ALIAS.
            # model_name_override is for the *actual* LLM model name.
            temp_provider_specific_kwargs = {"model_name": model_alias_override}
            if model_name_override: # If actual model name is also overridden
                temp_provider_specific_kwargs["model_name_override"] = model_name_override
            
            # Include other relevant kwargs passed to create_llm if they are fields of the ProviderSpecificSettingsClass
            for k, v in kwargs.items():
                if k in ProviderSpecificSettingsClass.model_fields:
                    temp_provider_specific_kwargs[k] = v
            
            provider_level_data_for_init = {
                k: v for k, v in provider_yaml_configs_raw.items() 
                if k not in ["models", "default_model_name"]
            }
            # Add global YAML settings that might be relevant if not in provider_level_data_for_init
            # (e.g., a global model_api_key if not in provider_yaml_configs_raw)
            # This part can be complex; for now, primary source is provider_yaml_configs_raw for provider-level data.

            try:
                provider_settings_obj = ProviderSpecificSettingsClass(
                    provider_models_config=provider_models_yaml_raw,
                    provider_default_model_alias=provider_default_alias_yaml, # May not be used if model_alias_override is strong
                    global_model_alias=None, # Simpler here, assumes alias priority logic in init handles it
                    provider_specific_kwargs=temp_provider_specific_kwargs,
                    **provider_level_data_for_init
                )
                logger.info(f"Dynamically created settings for '{target_provider_name}' with alias '{model_alias_override}'. Effective model name: '{provider_settings_obj.model_name}'.")
            except Exception as e:
                logger.error(f"Failed to dynamically create provider settings for {target_provider_name} with alias {model_alias_override}: {e}", exc_info=True)
                raise ValueError(f"Could not configure settings for {target_provider_name} with alias {model_alias_override}.")

        elif provider_settings_obj is None: # Should not happen if target_provider_name is valid and settings initialized
             raise ValueError(f"Settings object for provider '{target_provider_name}' is unexpectedly None in LangChainSettings.")

        # 4. Determine the final model name to use for the LLM
        # model_name_override directly sets the LLM model string (e.g., "gpt-4o-mini")
        # Otherwise, use the model_name resolved by the provider_settings_obj (based on its alias)
        final_model_name_for_llm = model_name_override if model_name_override is not None else provider_settings_obj.model_name
        
        if not final_model_name_for_llm:
            raise ValueError(
                f"Could not determine the final model name for provider '{target_provider_name}'. "
                f"Alias used: '{provider_settings_obj.model_alias}'. "
                "Ensure 'model_name' is set in YAML for this alias, or use 'model_name_override'."
            )

        # 5. Get the LLMProvider instance (e.g., OpenAIProvider instance)
        llm_provider_instance = LLMFactory.get_provider_instance(target_provider_name)

        # 6. Create the model
        # Pass only kwargs that are not already handled by provider_settings_obj fields
        # or are specific LLM creation params not part of settings.
        llm_creation_kwargs = kwargs.copy()
        # Remove keys that were used to initialize/select the provider_settings_obj if they are not direct LLM params
        # For example, model_alias_override and model_name_override are handled above.
        # However, kwargs like temperature, max_tokens, etc., if passed to create_llm,
        # should be passed to the actual LLM if they are valid for it.
        # The individual Provider.create_model methods can decide how to merge these with their settings.

        logger.info(f"Creating LLM: Provider='{target_provider_name}', Effective ModelName='{final_model_name_for_llm}', Alias='{provider_settings_obj.model_alias}'")

        llm_instance = llm_provider_instance.create_model(
            model_name_to_use=final_model_name_for_llm,
            provider_settings=provider_settings_obj,
            llm_specific_kwargs=llm_creation_kwargs 
        )
        
        return CreatedLLMInfo(
            llm_instance=llm_instance,
            provider_name=target_provider_name,
            model_name=final_model_name_for_llm,
            model_alias=provider_settings_obj.model_alias
        )

# Example of how a new provider would be automatically discovered:
# class MyNewProvider(LLMProvider):
#     def __init__(self):
#         super().__init__()
#         self.model_provider_name = self.get_provider_name()

#     def set_base_url(self, base_url: Optional[str]):
#         self.base_url = base_url
#         logger.info(f"MyNewProvider base_url set to: {base_url}")

#     def create_model(self, **kwargs) -> BaseChatModel:
#         # Implementation for creating model from MyNewProvider
#         logger.info(f"MyNewProvider creating model with {kwargs}")
#         # Replace with actual model creation logic
#         class DummyChatModel(BaseChatModel): # Dummy model for example
#             def _generate(self, messages, stop=None, run_manager=None, **kwargs): pass
#             async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs): pass
#             @property
#             def _llm_type(self) -> str: return "dummy-chat-model"

#         return DummyChatModel()

# To ensure discovery, the module containing MyNewProvider must be imported somewhere
# before LLMFactory.get_provider() is called for 'mynew'.