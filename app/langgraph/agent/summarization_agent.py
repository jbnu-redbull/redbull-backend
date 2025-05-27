from typing import Any, Dict
from ..langchain.base_chain import ChainOutput
from ..base_agent import BaseAgent
from ..response.summarization_model import SummarizationResponse

class SummarizationAgent(BaseAgent[ChainOutput]):
    """
    Summarization Agent
    - Receives a text and generates a concise summary.
    """

    agent_type: str = "summarization"
    response_model = SummarizationResponse
    default_prompt_template: str = """
Summarize the following text into a concise form in the same language.
Text: {text}

Respond strictly in JSON format like this: {format_instructions}
    """

    def summarize(self, text: str) -> SummarizationResponse:
        """
        Generates a summary for the given text.
        """
        input_data = {
            "text": text
        }
        result = self.invoke(input_data)
        if isinstance(result, SummarizationResponse):
            return result
        else:
            # 파서 문제 또는 LLM 응답 형식 불일치 시 예외 처리
            raise ValueError(f"Unexpected response type: {type(result)}. Expected SummarizationResponse.")
