import yaml
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator, PrivateAttr
from typing import Literal, Dict, Any, Optional, Union, Type, get_args
import logging
import datetime # For timestamping logs
import traceback # For detailed exception logging

logger = logging.getLogger(__name__)

# 프로젝트 루트 경로를 현재 파일 위치 기준으로 설정
# settings.py -> langchain -> langgraph -> app -> <project_root>
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

def load_yaml_config(file_path: Path) -> Dict[str, Any]:
    if file_path.exists() and file_path.is_file():
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                config_data = yaml.safe_load(f)
                return config_data if isinstance(config_data, dict) else {}
            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML file {file_path}: {e}")
                return {}
    return {}

# --- 개별 설정 클래스 정의 ---

class RetrySettings(BaseSettings):
    retry_max_attempts: int = Field(3, description="Maximum number of retry attempts.")
    retry_wait_multiplier: int = Field(1, description="Multiplier for exponential backoff wait time.")
    retry_min_interval_seconds: int = Field(1, description="Minimum wait interval in seconds for retries.")
    retry_max_interval_seconds: int = Field(60, description="Maximum wait interval in seconds for retries.")

    model_config = SettingsConfigDict(extra="ignore", validate_default=True)

class BaseProviderLLMSettings(BaseSettings):
    model_alias: Optional[str] = None # YAML에서 모델 설정을 찾기 위한 별칭
    model_name: Optional[str] = None  # 실제 LLM 모델 식별자
    model_temperature: Optional[float] = None
    model_max_tokens: Optional[int] = None
    model_timeout: Optional[int] = None
    
    # provider_all_config는 이 클래스를 상속받는 각 ProviderSettings에서 초기화 시 주입받음
    # 이를 통해 model_alias에 해당하는 모델의 상세 설정을 로드하는데 사용
    _provider_models_config: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _provider_default_model_alias: Optional[str] = PrivateAttr(default=None)
    _global_model_alias: Optional[str] = PrivateAttr(default=None)
    _kwargs_for_provider: Dict[str, Any] = PrivateAttr(default_factory=dict)

    model_config = SettingsConfigDict(extra="allow", validate_default=False) # extra='allow' 로 공급자별 추가 필드 허용

    def __init__(self, 
                 provider_models_config: Dict[str, Any], 
                 provider_default_model_alias: Optional[str],
                 global_model_alias: Optional[str],
                 provider_specific_kwargs: Dict[str, Any], # 이 Provider에 해당하는 kwargs만 필터링해서 받음
                 **provider_level_data: Any): # YAML의 provider 레벨 데이터 (models 제외)
        
        provider_class_name = self.__class__.__name__

        # 기존 print문 유지 (stdout이 잘리지 않는 환경에서는 유용할 수 있음)
        print(f"DEBUG BaseProvider.__init__ ({provider_class_name}): provider_specific_kwargs={provider_specific_kwargs}, provider_level_data={provider_level_data}")

        self._provider_models_config = provider_models_config
        self._provider_default_model_alias = provider_default_model_alias
        self._global_model_alias = global_model_alias
        self._kwargs_for_provider = provider_specific_kwargs

        try:
            # 기존 print문 유지
            print(f"DEBUG BaseProvider.__init__ ({provider_class_name}): About to call super().__init__ with {provider_level_data}")
            
            super().__init__(**provider_level_data) # API 키, URL 등 프로바이더 레벨 설정 초기화

            # 기존 print문 유지
            print(f"DEBUG BaseProvider.__init__ ({provider_class_name}): super().__init__ called successfully")

        except Exception as e_super:
            # 해당 예외를 다시 발생시켜서 LangChainSettings의 except 블록에서 처리되도록 함
            raise

        # 이제 model_alias, model_name 및 기타 모델 파라미터 결정
        # 우선순위: kwargs.model_name (alias로) > provider.default_model_name > global.model_name (alias로)
        # kwargs.model_name이 실제 LLM 모델명 override로 쓰이려면 명칭 변경 필요 (예: model_name_override)
        
        # 1. 사용할 모델 "별칭" 결정
        # kwargs는 이미 provider_specific_kwargs로 필터링 되어있으므로, model_name이 있다면 이 provider를 위한 것
        # kwargs의 model_name은 alias로 취급
        final_model_alias = provider_specific_kwargs.get("model_name", 
                                self._provider_default_model_alias or self._global_model_alias)
        self.model_alias = final_model_alias

        # 2. 결정된 모델 "별칭"에 해당하는 상세 설정 로드
        model_specific_config_yaml = {}
        if self.model_alias and self._provider_models_config:
            model_specific_config_yaml = self._provider_models_config.get(self.model_alias, {})

        # 3. 최종 설정 값 할당 (우선순위: kwargs > model_yaml > provider_yaml(super init에서 이미 일부 적용) > Pydantic 기본값 없음)
        
        # 실제 LLM 모델명 (model_name)
        # provider_specific_kwargs에 model_name_override가 있으면 최우선, 아니면 model_specific_config_yaml, 그것도 아니면 model_alias
        self.model_name = provider_specific_kwargs.get("model_name_override",
                                model_specific_config_yaml.get("model_name", self.model_alias))

        llm_params = ["model_temperature", "model_max_tokens", "model_timeout"]
        for param in llm_params:
            # kwargs 우선, 그 다음 model_specific_yaml, 그 다음 provider_level_data (super()에서 이미 설정됨)
            kwarg_val = provider_specific_kwargs.get(param)
            model_yaml_val = model_specific_config_yaml.get(param)
            
            if kwarg_val is not None:
                setattr(self, param, kwarg_val)
            elif model_yaml_val is not None:
                setattr(self, param, model_yaml_val)
            # else: provider_level_data 에서 이미 설정되었거나, None으로 유지

        # 공급자별 추가 필드들을 kwargs 또는 model_specific_config_yaml 에서 로드
        # BaseProviderLLMSettings에 명시적으로 정의되지 않은 필드들
        for key, value in model_specific_config_yaml.items():
            if not hasattr(self, key): # 이미 처리된 필드가 아니면
                setattr(self, key, value)
        for key, value in provider_specific_kwargs.items(): # kwargs가 최우선
             if not hasattr(self, key) or key in provider_specific_kwargs: # 이미 있더라도 kwargs면 덮어쓰기
                # model_name, model_name_override 등 이미 위에서 처리된 키는 제외
                if key not in ["model_name", "model_name_override"] + llm_params:
                    setattr(self, key, value)


class OpenAISettings(BaseProviderLLMSettings):
    model_provider: Literal["openai"] = "openai"
    model_api_key: Optional[str] = None
    # OpenAI 고유의 다른 설정들 추가 가능 (예: organization_id)

class AnthropicSettings(BaseProviderLLMSettings):
    model_provider: Literal["anthropic"] = "anthropic"
    model_api_key: Optional[str] = None
    # Anthropic 고유의 다른 설정들 추가 가능

class OllamaSettings(BaseProviderLLMSettings):
    model_provider: Literal["ollama"] = "ollama"
    model_provider_url: Optional[str] = None # Ollama는 API 키 대신 URL을 사용
    # Ollama 고유의 다른 설정들 추가 가능

class VLLMSettings(BaseProviderLLMSettings):
    model_provider: Literal["vllm"] = "vllm"
    model_provider_url: Optional[str] = None # vLLM 서버의 base URL
    model_api_key: Optional[str] = None # vLLM 서버가 API 키를 요구하는 경우 (OpenAI 호환)
    # vLLM의 OpenAI 호환 엔드포인트에서 지원하는 다른 파라미터가 있다면 여기에 추가 가능


PROVIDER_SETTINGS_MAP: Dict[str, Type[BaseProviderLLMSettings]] = {
    "openai": OpenAISettings,
    "anthropic": AnthropicSettings,
    "ollama": OllamaSettings,
    "vllm": VLLMSettings, # vLLM 설정 클래스 추가
}

# --- 메인 LangChainSettings 클래스 ---

class LangChainSettings(BaseSettings):
    active_model_provider: Optional[Literal["openai", "anthropic", "ollama", "vllm"]] = None # vllm 추가
    
    retry_settings: RetrySettings = Field(default_factory=RetrySettings)
    
    openai: Optional[OpenAISettings] = None
    anthropic: Optional[AnthropicSettings] = None
    ollama: Optional[OllamaSettings] = None
    vllm: Optional[VLLMSettings] = None # vllm 설정 필드 추가
    
    provider_configurations: Dict[str, Any] = Field(default_factory=dict, 
                                                     description="config.yaml의 llm_providers 전체 내용을 담는 딕셔너리")

    model_config = SettingsConfigDict(extra="ignore", validate_default=True)

    def __init__(self, **kwargs: Any):
        super().__init__() # retry_settings 등 Pydantic 기본값 적용 (여기서는 retry_settings만)

        # 1. YAML 파일 로드 및 병합 (이전과 동일한 로직 사용)
        local_config_path = Path(__file__).resolve().parent / "config.yaml"
        root_config_path = PROJECT_ROOT / "config.yaml"
        local_yaml_llm_settings = load_yaml_config(local_config_path).get("llm_settings", {})
        root_yaml_llm_settings = load_yaml_config(root_config_path).get("llm_settings", {})
        
        merged_yaml_llm_settings = local_yaml_llm_settings.copy()
        for key, value in root_yaml_llm_settings.items():
            if key == "llm_providers" and isinstance(value, dict) and isinstance(merged_yaml_llm_settings.get(key), dict):
                merged_providers = merged_yaml_llm_settings[key].copy()
                for prov_key, prov_value in value.items():
                    prov_existing_value = merged_providers.get(prov_key, {})
                    if isinstance(prov_value, dict) and isinstance(prov_existing_value, dict):
                        merged_provider_details = prov_existing_value.copy()
                        # 모델 병합
                        models_yaml = prov_value.get("models", {})
                        existing_models_yaml = merged_provider_details.get("models", {})
                        merged_models = existing_models_yaml.copy()
                        for model_k, model_v in models_yaml.items():
                            merged_models[model_k] = {**existing_models_yaml.get(model_k, {}), **model_v}
                        if merged_models: merged_provider_details["models"] = merged_models
                        
                        # 모델 외 다른 프로바이더 레벨 키 병합
                        for k,v in prov_value.items():
                            if k != "models": merged_provider_details[k] = v
                        merged_providers[prov_key] = merged_provider_details
                    else:
                        merged_providers[prov_key] = prov_value
                merged_yaml_llm_settings[key] = merged_providers
            else:
                merged_yaml_llm_settings[key] = value
        
        self.provider_configurations = merged_yaml_llm_settings.get("llm_providers", {})

        # 2. RetrySettings 초기화 (YAML 전역 설정 -> kwargs 순으로 덮어쓰기)
        yaml_global_retry_settings = merged_yaml_llm_settings.get("retry_settings", {})
        # kwargs에서 retry 관련 설정만 필터링
        retry_kwargs = {k: v for k, v in kwargs.items() if k.startswith("retry_")}
        self.retry_settings = RetrySettings(**{**yaml_global_retry_settings, **retry_kwargs})

        # 3. Active model provider 결정 (kwargs > YAML 전역)
        self.active_model_provider = kwargs.get("active_model_provider", 
                                              kwargs.get("model_provider", # 이전 호환성
                                                         merged_yaml_llm_settings.get("model_provider")))
        if not self.active_model_provider:
            raise ValueError(
                "active_model_provider must be set either via LangChainSettings constructor (kwargs) "
                "or in config.yaml (llm_settings.model_provider)."
            )
        if self.active_model_provider not in PROVIDER_SETTINGS_MAP:
            raise ValueError(
                f"Invalid active_model_provider: '{self.active_model_provider}'. "
                f"Allowed values are: {list(PROVIDER_SETTINGS_MAP.keys())}"
            )
            
        # 4. 각 공급자별 설정 객체 초기화
        global_model_alias_from_yaml = merged_yaml_llm_settings.get("model_name") # 전역 model_name은 alias로

        for provider_name, ProviderSettingClass in PROVIDER_SETTINGS_MAP.items():
            # --->>> LangChainSettings: Processing provider_name = ollama <<<---
            # ... (기존 추가했던 print 문들) ...
            print(f"--->>> LangChainSettings: Processing provider_name = {provider_name} <<<---")
            provider_data_yaml = self.provider_configurations.get(provider_name, {})
            
            provider_specific_kwargs = {}
            for kw_key, kw_value in kwargs.items():
                if kw_key.startswith(provider_name + "_"):
                    provider_specific_kwargs[kw_key.split(provider_name + "_", 1)[1]] = kw_value
                elif not any(kw_key.startswith(p + "_") for p in PROVIDER_SETTINGS_MAP if p != provider_name) and \
                     kw_key not in ["active_model_provider", "model_provider"] and not kw_key.startswith("retry_") :
                    if provider_name == self.active_model_provider:
                         provider_specific_kwargs[kw_key] = kw_value

            provider_models_config = provider_data_yaml.get("models", {})
            provider_default_model_alias = provider_data_yaml.get("default_model_name")
            
            provider_level_data_for_init = {
                k: v for k, v in provider_data_yaml.items()
                if k not in ["models", "default_model_name"]
            }

            # 전역 설정 중 공급자별 설정 클래스 필드와 일치하는 것들 (예: model_api_key)을 기본값으로 추가
            # 단, provider_level_data_for_init에 이미 있으면 YAML 프로바이더 레벨이 우선
            for field_name in ProviderSettingClass.model_fields.keys():
                # 'model_provider'는 각 ProviderSettingClass에 이미 Literal로 정의되어 있으므로,
                # 전역 설정에서 덮어쓰지 않도록 제외합니다.
                if field_name == "model_provider":
                    continue
                if field_name in merged_yaml_llm_settings and field_name not in provider_level_data_for_init:
                    provider_level_data_for_init[field_name] = merged_yaml_llm_settings[field_name]

            print(f"--->>> LangChainSettings: Data for {provider_name} before calling its constructor:")
            print(f"        ProviderSettingClass: {ProviderSettingClass.__name__}")
            print(f"        provider_models_config: {provider_models_config}")
            print(f"        provider_default_model_alias: {provider_default_model_alias}")
            print(f"        global_model_alias: {global_model_alias_from_yaml}")
            print(f"        provider_specific_kwargs: {provider_specific_kwargs}")
            print(f"        provider_level_data_for_init: {provider_level_data_for_init}")

            try:
                instance = ProviderSettingClass(
                    provider_models_config=provider_models_config,
                    provider_default_model_alias=provider_default_model_alias,
                    global_model_alias=global_model_alias_from_yaml,
                    provider_specific_kwargs=provider_specific_kwargs,
                    **provider_level_data_for_init
                )
                setattr(self, provider_name, instance)
                print(f"--->>> LangChainSettings: Successfully initialized {provider_name} <<<---")
            except Exception as e:
                print(f"--->>> LangChainSettings EXCEPTION for {provider_name}: {type(e).__name__} - {str(e)} <<<---")
                logger.error(f"Failed to initialize settings for provider {provider_name}: {e}", exc_info=True)
                setattr(self, provider_name, None)

    @property
    def current_llm_settings(self) -> Optional[BaseProviderLLMSettings]:
        if self.active_model_provider:
            return getattr(self, self.active_model_provider.lower(), None)
        return None

    # 편의를 위한 현재 활성 LLM의 주요 속성 접근자
    @property
    def model_provider(self) -> Optional[str]:
        return self.active_model_provider

    @property
    def model_name(self) -> Optional[str]:
        return self.current_llm_settings.model_name if self.current_llm_settings else None

    @property
    def model_alias(self) -> Optional[str]:
        return self.current_llm_settings.model_alias if self.current_llm_settings else None
        
    @property
    def model_api_key(self) -> Optional[str]:
        if self.current_llm_settings and hasattr(self.current_llm_settings, 'model_api_key'):
            return self.current_llm_settings.model_api_key
        return None

    @property
    def model_provider_url(self) -> Optional[str]:
        if self.current_llm_settings and hasattr(self.current_llm_settings, 'model_provider_url'):
            return self.current_llm_settings.model_provider_url
        return None

    @property
    def model_temperature(self) -> Optional[float]:
        return self.current_llm_settings.model_temperature if self.current_llm_settings else None

    @property
    def model_max_tokens(self) -> Optional[int]:
        return self.current_llm_settings.model_max_tokens if self.current_llm_settings else None

    @property
    def model_timeout(self) -> Optional[int]:
        return self.current_llm_settings.model_timeout if self.current_llm_settings else None

    # 다른 필요한 속성들도 유사하게 추가 가능

    @model_validator(mode='after')
    def check_active_provider_settings_initialized(self) -> 'LangChainSettings':
        if self.active_model_provider:
            if self.current_llm_settings is None:
                logger.warning(
                    f"Settings for active provider '{self.active_model_provider}' "
                    "could not be initialized. Check YAML configuration and constructor arguments."
                )
            # elif self.current_llm_settings.model_name is None: # 이제 model_name은 alias가 될수도 있음
            #     logger.warning(
            #         f"For active provider '{self.active_model_provider}', 'model_name' is not set. "
            #         "Ensure it's available via 'model_name_override' in kwargs, or "
            #         "'model_name' in YAML model config, or as a fallback from 'model_alias'."
            #     )
        return self


# 전역 설정 인스턴스
try:
    print("[settings.py] Attempting to initialize LangChainSettings...")
    langchain_settings = LangChainSettings()
    print("[settings.py] LangChainSettings initialized.")
    # 초기화 후 로그 기록
    if langchain_settings and langchain_settings.current_llm_settings:
        print("[settings.py] langchain_settings and current_llm_settings are valid. Attempting logger.info and logger.debug...")
        logger.info(f"Default LangChainSettings initialized for provider: {langchain_settings.active_model_provider}")
        logger.debug(f"Active LLM Settings: Provider='{langchain_settings.model_provider}', "
                     f"Alias='{langchain_settings.model_alias}', Name='{langchain_settings.model_name}', "
                     f"APIKeySet={langchain_settings.model_api_key is not None}, "
                     f"Temp='{langchain_settings.model_temperature}', URL='{langchain_settings.model_provider_url}'")
        print("[settings.py] logger.info and logger.debug attempted.")
    elif langchain_settings:
         print("[settings.py] langchain_settings is valid but current_llm_settings is not. Attempting logger.warning...")
         logger.warning(f"Default LangChainSettings initialized, but no current LLM settings for active provider: {langchain_settings.active_model_provider}")
         print("[settings.py] logger.warning attempted.")
    else:
        print("[settings.py] langchain_settings is None after initialization attempt (should not happen here).")

except ValueError as e:
    print(f"[settings.py] CRITICAL - ValueError during LangChainSettings initialization: {e}")
    logger.error(f"CRITICAL: Failed to initialize default LangChainSettings due to ValueError: {e}")
    langchain_settings = None 
except Exception as e:
    print(f"[settings.py] CRITICAL - Exception during LangChainSettings initialization: {e}")
    logger.error(f"CRITICAL: An unexpected error occurred during LangChainSettings initialization: {e}", exc_info=True)
    langchain_settings = None


def set_langchain_settings(_langchain_settings: LangChainSettings):
    global langchain_settings
    langchain_settings = _langchain_settings
    logger.info(f"LangChainSettings have been explicitly set. Active provider: {langchain_settings.active_model_provider if langchain_settings else 'None'}")
    return langchain_settings