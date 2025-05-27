from typing import Type, Optional, Dict, Any
# 경로 수정: langchain 폴더 내부의 base_chain 및 settings를 참조하도록 변경
from .langchain.base_chain import BaseChain, ChainOutput
from .langchain.settings import langchain_settings, LangChainSettings
# BaseAgent 임포트 추가
from .base_agent import BaseAgent
# 기존 경로 유지 (translation_agent는 app/langgraph/ 하위에 직접 위치)
from .agent.translation_agent import TranslationAgent
# 경로 수정: TranslationResponse를 app.langgraph.response.translation_model에서 가져옴
from .agent.response.translation_model import TranslationResponse
# 다른 에이전트 및 응답 모델을 필요에 따라 임포트합니다.
# from app.langgraph.another_agent import AnotherAgent
# from app.langgraph.response.another_model import AnotherResponse # 경로 변경 가능성
from pydantic import BaseModel

import logging
import importlib
import inspect
# import os # 이미 주석 처리됨 또는 제거됨
import pkgutil

logger = logging.getLogger(__name__)

class AgentFactoryError(Exception):
    """Custom exception for AgentFactory errors."""
    pass

class AgentFactory:
    """    Factory class for creating instances of different agents.
    This factory helps in centralizing the creation logic of agents
    and managing their dependencies.
    """

    AGENT_REGISTRY: Dict[str, Dict[str, Any]] = {} # 동적으로 채워짐

    @classmethod
    def register_agent(cls, agent_class: Type[BaseAgent]):
        """Registers an agent class in the factory's registry if it's a valid BaseAgent subclass."""
        # BaseAgent 자체는 등록하지 않음
        if not inspect.isclass(agent_class) or not issubclass(agent_class, BaseAgent) or agent_class is BaseAgent:
            # logger.debug(f"Skipping registration for {agent_class_name}: Not a valid subclass of BaseAgent.")
            return

        agent_type_val = getattr(agent_class, 'agent_type', None)
        agent_class_name = agent_class.__name__

        if not agent_type_val:
            logger.warning(f"Agent class {agent_class_name} does not have an 'agent_type' attribute. Skipping registration.")
            return

        response_model_val = getattr(agent_class, 'response_model', None)
        if not response_model_val or not issubclass(response_model_val, BaseModel):
            logger.warning(f"Agent class {agent_class_name} (type: {agent_type_val}) does not have a valid 'response_model' (Pydantic BaseModel). Skipping registration.")
            return
        
        default_prompt_val = getattr(agent_class, 'default_prompt_template', None)
        # default_prompt_template은 Optional이므로 없어도 경고 없이 등록

        if agent_type_val in cls.AGENT_REGISTRY:
            # 동일한 agent_type으로 다른 클래스가 이미 등록된 경우 경고. 동일 클래스면 무시.
            if cls.AGENT_REGISTRY[agent_type_val]["agent_class"] != agent_class:
                logger.warning(f"Agent type '{agent_type_val}' is already registered with class {cls.AGENT_REGISTRY[agent_type_val]['agent_class'].__name__}. "
                                 f"Overwriting with {agent_class_name}.")
            # else: # 동일 클래스면 재등록할 필요 없음
            #     return

        cls.AGENT_REGISTRY[agent_type_val] = {
            "agent_class": agent_class,
            "response_model": response_model_val,
            "default_prompt_template": default_prompt_val
        }
        logger.info(f"Successfully registered agent: {agent_type_val} -> {agent_class_name}")

    @classmethod
    def discover_agents(cls, package_name: str = "app.langgraph"):
        """
        Discovers and registers agent classes from the specified Python package.
        It looks for subclasses of BaseAgent.
        This method should be called once at application startup.
        """
        if cls.AGENT_REGISTRY: # 이미 탐색/등록이 완료되었다면 중복 실행 방지
            logger.info("Agent discovery has already been performed. Skipping.")
            # return # 필요에 따라 재탐색을 허용할 수도 있음

        logger.info(f"Starting agent discovery in package: {package_name}")
        try:
            package = importlib.import_module(package_name)
            
            # 패키지 내의 모든 모듈을 순회 (하위 패키지 포함하여 탐색)
            for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                try:
                    module = importlib.import_module(module_info.name)
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and issubclass(obj, BaseAgent) and obj is not BaseAgent:
                            # register_agent 내부에서 중복 등록 및 유효성 검사 수행
                            cls.register_agent(obj)
                except ImportError as e:
                    logger.error(f"Could not import module {module_info.name} during agent discovery: {e}")
                except Exception as e:
                    logger.error(f"Error inspecting module {module_info.name} during agent discovery: {e}", exc_info=True)
            
            logger.info(f"Agent discovery completed. Registered agents: {list(cls.AGENT_REGISTRY.keys())}")
            if not cls.AGENT_REGISTRY:
                logger.warning("No agents were discovered or registered. Ensure BaseAgent subclasses exist in the specified package and have 'agent_type' and 'response_model'.")

        except ImportError as e:
            logger.error(f"Could not import base package {package_name} for agent discovery: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during agent discovery in {package_name}: {e}", exc_info=True)

    @staticmethod
    def create_agent(
        agent_type: str,
        langchain_settings_obj: LangChainSettings, 
        prompt_template_str_override: Optional[str] = None, 
        provider_name_override: Optional[str] = None,    
        model_alias_override: Optional[str] = None,      
        **chain_kwargs: Any 
    ) -> BaseAgent[ChainOutput]: 
        
        if not AgentFactory.AGENT_REGISTRY:
            logger.warning("Agent registry empty. Performing initial discovery.")
            AgentFactory.discover_agents()

        logger.info(
            f"Attempting to create agent: {agent_type}, "
            f"provider_override='{provider_name_override}', model_alias_override='{model_alias_override}'"
        )

        if agent_type not in AgentFactory.AGENT_REGISTRY:
            logger.error(f"Agent type '{agent_type}' not recognized. Available: {list(AgentFactory.AGENT_REGISTRY.keys())}")
            logger.info("Attempting re-discovery as fallback...")
            AgentFactory.discover_agents()
            if agent_type not in AgentFactory.AGENT_REGISTRY:
                 raise AgentFactoryError(f"Agent type '{agent_type}' still not recognized after re-discovery.")

        agent_config = AgentFactory.AGENT_REGISTRY[agent_type]
        agent_class: Type[BaseAgent] = agent_config["agent_class"]
        # response_model과 default_prompt_template은 BaseAgent의 __init__에서 클래스 변수로부터 가져와 사용함.
        # create_agent 호출 시 prompt_template이 명시적으로 제공되면 해당 값을 사용하고,
        # 그렇지 않으면 agent_class에 정의된 default_prompt_template이 BaseAgent의 __init__ 내부에서 사용됨.

        try:
            logger.debug(f"Creating agent '{agent_type}' with class '{agent_class.__name__}'")
            agent_instance = agent_class(
                langchain_settings_obj=langchain_settings_obj,
                prompt_template_str_override=prompt_template_str_override,
                provider_name_override=provider_name_override,
                model_alias_override=model_alias_override,
                **chain_kwargs
            )
            logger.info(f"Agent '{agent_type}' (class: {agent_class.__name__}) created successfully.")
            return agent_instance
        except ValueError as ve: # BaseAgent의 __init__에서 발생할 수 있는 ValueError 처리
            logger.error(f"Failed to create agent '{agent_type}' due to invalid configuration: {ve}", exc_info=True)
            raise AgentFactoryError(f"Could not create agent '{agent_type}' due to invalid configuration: {ve}")
        except Exception as e:
            logger.error(f"Failed to create agent '{agent_type}': {e}", exc_info=True)
            raise AgentFactoryError(f"Could not create agent '{agent_type}': {e}")

# 애플리케이션 시작 시점에 에이전트들을 탐색하도록 권장.
# 예: main.py 또는 app 초기화 파일에서 AgentFactory.discover_agents() 호출
# 이 파일이 직접 실행될 때만 동작하는 __main__ 블록에 두는 것은 일반적인 사용 사례에 적합하지 않음.

# Example usage (for testing if this file is run directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 테스트 실행 시 에이전트 탐색
    AgentFactory.discover_agents(package_name="app.langgraph")

    # TranslationResponse를 직접 임포트할 필요는 없어짐 (테스트 코드 내에서는)
    # from app.langgraph.response.translation_model import TranslationResponse
    # TranslationAgent도 직접 임포트할 필요는 없어짐
    # from app.langgraph.agent.translation_agent import TranslationAgent 

    if not langchain_settings.model_api_key:
        logger.warning("API Key not found in langchain_settings. Translation agent might not work or test will be skipped.")
    
    try:
        print("\n--- Creating Translation Agent (Default Settings via Factory) ---")
        if "translation" in AgentFactory.AGENT_REGISTRY:
            translation_agent = AgentFactory.create_agent(
                agent_type="translation", 
                langchain_settings_obj=langchain_settings
            )
            print(f"Successfully created agent: {type(translation_agent).__name__} of type '{translation_agent.agent_type}'")
            print(f"Effective LLM: {translation_agent.effective_llm_provider}/{translation_agent.effective_model_name}")

            if langchain_settings.model_api_key:
                try:
                    # TranslationAgent에는 translate 메서드가 있어야 함
                    if hasattr(translation_agent, 'atranslate') and callable(getattr(translation_agent, 'atranslate')):
                        # 타입 체커를 위해 ignore 추가 (create_agent는 BaseAgent를 반환하므로)
                        response = translation_agent.atranslate(text="Hello, how are you today?", target_language="Korean") # type: ignore
                        print(f"Translation successful: '{response.translated_text}'")
                        print(f"Detected source language: {response.detected_source_language}")
                    else:
                        logger.error("Created 'translation' agent does not have a callable 'atranslate' method.")
                except Exception as e:
                    print(f"Error during translation test: {e}", exc_info=True)
            else:
                print("Skipping translation test as API key is not configured in langchain_settings.")
        else:
            print("Translation agent type 'translation' not found in registry. Skipping test.")

        print("\n--- Creating Translation Agent (Custom Prompt via Factory) ---")
        if "translation" in AgentFactory.AGENT_REGISTRY:
            custom_prompt = '''Translate the following text into {target_language}.
Original text: {text}
Please make the translation sound very formal and use honorifics where appropriate.
Respond strictly in JSON format, like so: {format_instructions}'''
            
            custom_translation_agent = AgentFactory.create_agent(
                agent_type="translation",
                langchain_settings_obj=langchain_settings,
                prompt_template_str_override=custom_prompt,
                provider_name_override="openai", 
                model_alias_override="gpt-4o-mini" # config.yaml의 openai.models 하위 별칭
            )
            print(f"Successfully created agent with custom prompt: {type(custom_translation_agent).__name__}")
            print(f"Agent effective LLM: {custom_translation_agent.effective_llm_provider}/{custom_translation_agent.effective_model_name}")
            # 여기에 custom_translation_agent를 사용한 API 호출 테스트 추가 가능
        else:
            print("Translation agent type 'translation' not found in registry. Skipping custom prompt test.")


        print("\n--- Attempting to create non-existent agent (Factory) ---")
        try:
            AgentFactory.create_agent("non_existent_fancy_agent", langchain_settings, None, None, None)
        except AgentFactoryError as e:
            print(f"Caught expected error for non-existent agent: {e}")

    except AgentFactoryError as afe:
        print(f"AgentFactoryError during test: {afe}", exc_info=True)
    except Exception as ex:
        print(f"An unexpected error occurred during test: {ex}", exc_info=True) 