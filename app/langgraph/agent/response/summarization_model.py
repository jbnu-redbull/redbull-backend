from ..langchain.response.base_response import CustomResponseModel
from pydantic import Field

class SummarizationResponse(CustomResponseModel):
    summary: str = Field(..., description="The summarized text.")

    @classmethod
    def format_llm_instructions(cls, example: dict) -> str:
        return (
            "Please provide a JSON object that summarizes the text. "
            "Ensure your response matches the following format:\n\n"
            f"{example}"
        )
