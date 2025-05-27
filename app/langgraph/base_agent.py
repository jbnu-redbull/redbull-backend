from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Dict, Any, Optional, Type, AsyncGenerator
from pydantic import BaseModel
import logging # logging 임포트 추가

from .langchain.base_chain import BaseChain, ChainOutput, ChainError
from .langchain.settings import LangChainSettings # 명시적 임포트

logger = logging.getLogger(__name__) # 로거 초기화

# AgentFactory 클래스를 여기서 직접 참조하는 대신, AgentFactory에서 register_agent를 호출하도록 변경
# from app.langgraph.agent_factory import AgentFactory # 순환 참조 가능성 제거

RegisteredAgent = TypeVar('RegisteredAgent', bound='BaseAgent')

class BaseAgent(BaseChain[ChainOutput], ABC):
    """
    Abstract base class for all specific agents.
    It standardizes the agent's interface and integrates with the AgentFactory
    for automatic registration.
    """
    agent_type: str 
    response_model: Type[BaseModel] 
    default_prompt_template: Optional[str] = None

    # __init_subclass__ 로직은 AgentFactory.discover_agents()에 의해 처리되므로 불필요
    # def __init_subclass__(cls, **kwargs):
    #     super().__init_subclass__(**kwargs)

    def __init__(
        self,
        langchain_settings_obj: LangChainSettings, 
        prompt_template_str_override: Optional[str] = None, 
        provider_name_override: Optional[str] = None, 
        model_alias_override: Optional[str] = None,  
        **chain_kwargs: Any 
    ):
        cls_response_model = getattr(self.__class__, 'response_model', None)
        cls_default_prompt = getattr(self.__class__, 'default_prompt_template', None)
        cls_agent_type = getattr(self.__class__, 'agent_type', "UnknownAgent")

        final_prompt_template = prompt_template_str_override if prompt_template_str_override is not None else cls_default_prompt

        if final_prompt_template is None:
            raise ValueError(
                f"Prompt template must be provided via prompt_template_str_override or "
                f"defined in the agent class ({self.__class__.__name__}) via default_prompt_template."
            )
        
        if not cls_response_model or not issubclass(cls_response_model, BaseModel):
             raise ValueError(
                 f"response_model class variable must be defined and be a Pydantic BaseModel "
                 f"in the agent class ({self.__class__.__name__})."
             )

        super().__init__(
            langchain_settings_obj=langchain_settings_obj, 
            prompt_template_str=final_prompt_template,    
            output_response_model=cls_response_model,    
            provider_name_override=provider_name_override, 
            model_alias_override=model_alias_override,     
            **chain_kwargs 
        )
        
        logger.info(
            f"{self.__class__.__name__} (type: {cls_agent_type}) initialized. "
            f"Effective LLM: {self.effective_llm_provider}/{self.effective_model_name}"
        )

    async def astream(
        self,
        input_data: Dict[str, Any],
        **kwargs: Any 
    ) -> AsyncGenerator[Any, None]:
        """
        Streams responses from the underlying language model chain.
        This agent-level astream method calls the astream method of its BaseChain superclass.
        """
        logger.debug(f"Agent {self.__class__.__name__} astream calling super().astream with input: {input_data}, kwargs: {kwargs}")
        try:
            # BaseChain의 astream 메서드를 호출 (이것이 모킹 대상이 됨)
            # BaseChain.astream은 config 인자를 받을 수 있으므로, kwargs를 통해 전달되도록 함.
            async for chunk in super().astream(input_data, **kwargs):
                yield chunk
        except Exception as e:
            logger.error(f"Error during agent astream for {self.__class__.__name__}: {e}", exc_info=True)
            if hasattr(self, '_classify_error') and callable(getattr(self, '_classify_error')):
                raise self._classify_error(e)
            else:
                raise ChainError(f"An unexpected error occurred during agent astream: {str(e)}", original_exception=e)

    # 특정 에이전트의 핵심 로직 (예: translate)은 각 서브클래스에서 구현
    # 예시:
    # @abstractmethod
    # def execute(self, input_data: Dict[str, Any], **kwargs) -> ChainOutput:
    #     """Executes the core logic of the agent."""
    #     pass

