import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import os
import sys
import logging
# from dotenv import dotenv_values # .env 파일 파싱용 - 제거됨
import argparse # argparse 임포트
import json
from typing import Optional

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# 수정: AgentFactory 임포트
from app.langgraph.agent_factory import AgentFactory, AgentFactoryError
# TranslationAgent는 직접 임포트할 필요가 없을 수 있지만, isinstance 확인 등을 위해 유지 가능
from app.langgraph.agent.translation_agent import TranslationAgent 
# 수정: TranslationResponse 임포트 경로 변경
from app.langgraph.agent.response.translation_model import TranslationResponse 
from app.langgraph.langchain.base_chain import OutputParserException, ChainError
from app.langgraph.langchain.settings import LangChainSettings, set_langchain_settings, langchain_settings as initial_global_settings
from langchain_core.messages import AIMessageChunk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestTranslationAgent(unittest.IsolatedAsyncioTestCase):

    original_settings_dump: Optional[str] = None # 설정 복원을 위해 JSON 덤프 저장
    # agent_under_test = None # 제거 또는 필요시 사용

    @classmethod
    def setUpClass(cls):
        # __main__에서 CLI 인자가 적용된 후의 전역 settings를 가져옴
        from app.langgraph.langchain.settings import langchain_settings as current_settings_after_cli_parse
        cls.original_settings_dump = current_settings_after_cli_parse.model_dump_json()
        logger.info(f"--- TestTranslationAgent.setUpClass: Using langchain_settings: {cls.original_settings_dump} ---")

        try:
            AgentFactory.AGENT_REGISTRY.clear()
            AgentFactory.discover_agents(package_name="app.langgraph")
            if "translation" not in AgentFactory.AGENT_REGISTRY:
                 logger.critical("Translation agent not found after discovery!")
        except Exception as e:
            logger.critical(f"Error during agent discovery in setUpClass: {e}", exc_info=True)

    @classmethod
    def tearDownClass(cls):
        # 저장된 원래 설정(JSON 덤프)으로 복원 시도
        # 이 방식은 set_langchain_settings가 전역 객체를 올바르게 교체한다고 가정
        if cls.original_settings_dump:
            try:
                original_settings_data = json.loads(cls.original_settings_dump)
                restored_settings = LangChainSettings(**original_settings_data)
                set_langchain_settings(restored_settings) # 전역 settings 객체를 복원된 것으로 교체
                logger.info(f"--- TestTranslationAgent.tearDownClass: Restored langchain_settings to: {restored_settings.model_dump_json()} ---")
            except Exception as e:
                logger.error(f"Failed to restore langchain_settings in tearDownClass: {e}", exc_info=True)
        
        AgentFactory.AGENT_REGISTRY.clear()
        logger.info("Finished TranslationAgent tests. AgentFactory registry cleared.")

    def setUp(self):
        """Set up for each test method."""
        from app.langgraph.langchain.settings import langchain_settings as current_test_settings
        try:
            self.agent = AgentFactory.create_agent(
                agent_type="translation",
                langchain_settings_obj=current_test_settings 
            )
        except AgentFactoryError as e:
            logger.error(f"Failed to create translation agent in setUp: {e}", exc_info=True)
            self.fail(f"Failed to create translation agent in setUp: {e}")
            return # ایجنٹ نہیں بن سکا تو आगे نہ بڑھیں
            
        self.text_to_translate = "Hello, world!"
        self.target_language = "Korean"
        logger.info(
            f"TestTranslationAgent setUp: Agent initialized with model: "
            f"{self.agent.effective_llm_provider}/{self.agent.effective_model_name} "
            f"for test {self._testMethodName}"
        )

    def tearDown(self):
        logger.info(f"Finished test: {self._testMethodName}")
        print("\n" * 5) # 원래대로 복원 및 주석 제거

    @patch('app.langgraph.langchain.base_chain.BaseChain.invoke')
    def test_translate_successful(self, mock_invoke):
        """Test successful synchronous translation."""
        # self.agent가 setUp에서 성공적으로 생성되었다고 가정
        if not self.agent: self.skipTest("Agent not created in setUp")

        expected_response_data = {
            # "original_text": self.text_to_translate, # translate 메서드 내에서 채워짐
            "translated_text": "안녕하세요, 세계!",
            "target_language": self.target_language,
            "detected_source_language": "English",
            "confidence_score": 0.95
        }
        mock_invoke.return_value = TranslationResponse(**expected_response_data, original_text=self.text_to_translate)

        result = self.agent.translate(self.text_to_translate, self.target_language)

        self.assertIsInstance(result, TranslationResponse)
        self.assertEqual(result.original_text, self.text_to_translate)
        self.assertEqual(result.translated_text, expected_response_data["translated_text"])
        self.assertEqual(result.target_language, self.target_language)
        self.assertEqual(result.detected_source_language, expected_response_data["detected_source_language"])
        mock_invoke.assert_called_once_with({"text": self.text_to_translate, "target_language": self.target_language})

    @patch('app.langgraph.langchain.base_chain.BaseChain.ainvoke', new_callable=AsyncMock)
    async def test_atranslate_successful(self, mock_ainvoke):
        """Test successful asynchronous translation."""
        if not self.agent: self.skipTest("Agent not created in setUp")
        expected_response_data = {
            "translated_text": "안녕하세요, 세계!",
            "target_language": self.target_language,
            "detected_source_language": "English",
            "confidence_score": 0.9
        }
        mock_ainvoke.return_value = TranslationResponse(**expected_response_data, original_text=self.text_to_translate)

        result = await self.agent.atranslate(self.text_to_translate, self.target_language)

        self.assertIsInstance(result, TranslationResponse)
        self.assertEqual(result.original_text, self.text_to_translate)
        self.assertEqual(result.translated_text, expected_response_data["translated_text"])
        mock_ainvoke.assert_called_once_with({"text": self.text_to_translate, "target_language": self.target_language})

    @patch('app.langgraph.langchain.base_chain.BaseChain.invoke')
    def test_translate_output_parser_exception(self, mock_invoke):
        """Test OutputParserException during synchronous translation."""
        if not self.agent: self.skipTest("Agent not created in setUp")
        malformed_llm_output = "This is not JSON"
        mock_invoke.side_effect = OutputParserException(
            "Failed to parse",
            llm_output=malformed_llm_output
        )

        with self.assertRaises(OutputParserException) as context:
            self.agent.translate(self.text_to_translate, self.target_language)
        
        self.assertIn("Failed to parse", str(context.exception))
        mock_invoke.assert_called_once()

    @patch('app.langgraph.langchain.base_chain.BaseChain.ainvoke', new_callable=AsyncMock)
    async def test_atranslate_output_parser_exception(self, mock_ainvoke):
        """Test OutputParserException during asynchronous translation."""
        if not self.agent: self.skipTest("Agent not created in setUp")
        malformed_llm_output = "Still not JSON"
        mock_ainvoke.side_effect = OutputParserException(
            "Async parse failed",
            llm_output=malformed_llm_output 
        )

        with self.assertRaises(OutputParserException) as context:
            await self.agent.atranslate(self.text_to_translate, self.target_language)
        
        self.assertIn("Async parse failed", str(context.exception))
        mock_ainvoke.assert_called_once()

    @patch('app.langgraph.langchain.base_chain.BaseChain.invoke')
    def test_translate_chain_error(self, mock_invoke):
        if not self.agent: self.skipTest("Agent not created in setUp")
        mock_invoke.side_effect = ChainError("LLM unavailable")
        with self.assertRaises(ChainError) as context:
            self.agent.translate(self.text_to_translate, self.target_language)
        self.assertIn("LLM unavailable", str(context.exception))
        mock_invoke.assert_called_once()

    @patch('app.langgraph.langchain.base_chain.BaseChain.ainvoke', new_callable=AsyncMock)
    async def test_atranslate_chain_error(self, mock_ainvoke):
        if not self.agent: self.skipTest("Agent not created in setUp")
        mock_ainvoke.side_effect = ChainError("Async LLM unavailable")
        with self.assertRaises(ChainError) as context:
            await self.agent.atranslate(self.text_to_translate, self.target_language)
        self.assertIn("Async LLM unavailable", str(context.exception))
        mock_ainvoke.assert_called_once()

    @patch('app.langgraph.langchain.base_chain.BaseChain.astream', new_callable=MagicMock)
    async def test_astream_successful(self, mock_astream_base_chain):
        if not self.agent: self.skipTest("Agent not created in setUp")

        expected_response_data = {
            "original_text": self.text_to_translate,
            "translated_text": "안녕하세요, 세계!",
            "target_language": self.target_language,
            "detected_source_language": "English",
            "confidence_score": 0.95
        }
        expected_translation_response = TranslationResponse(**expected_response_data)

        # This inner async def function, when called, returns an async generator object.
        async def mock_response_generator_impl():
            yield expected_translation_response
        
        # Set the return_value of the MagicMock to an actual async generator object.
        mock_astream_base_chain.return_value = mock_response_generator_impl()
        
        input_data = {"text": self.text_to_translate, "target_language": self.target_language}
        
        received_responses = []
        # self.agent.astream은 BaseAgent.astream을 호출하고, 
        # 이는 내부적으로 BaseChain.astream (여기서는 mock_astream_base_chain)을 호출
        async for response_obj in self.agent.astream(input_data):
            received_responses.append(response_obj)
        
        self.assertEqual(len(received_responses), 1, "Should have received exactly one response object.")
        actual_response: TranslationResponse = received_responses[0]
        
        self.assertIsInstance(actual_response, TranslationResponse)
        self.assertEqual(actual_response.original_text, expected_translation_response.original_text)
        self.assertEqual(actual_response.translated_text, expected_translation_response.translated_text)
        self.assertEqual(actual_response.target_language, expected_translation_response.target_language)
        
        # BaseChain.astream이 올바른 입력으로 호출되었는지 확인
        mock_astream_base_chain.assert_called_once_with(input_data)

    @patch('app.langgraph.langchain.base_chain.BaseChain.stream')
    def test_stream_successful(self, mock_stream):
        if not self.agent: self.skipTest("Agent not created in setUp")
        def mock_stream_generator():
            yield AIMessageChunk(content="{\n")
            yield AIMessageChunk(content="  \"translated_text\": \"안녕하세요")
            yield AIMessageChunk(content=", 세계!\",\n")
            yield AIMessageChunk(content="  \"target_language\": \"Korean\",\n")
            yield AIMessageChunk(content="  \"detected_source_language\": \"English\"\n")
            yield AIMessageChunk(content="}")

        mock_stream.return_value = mock_stream_generator()
        input_data = {"text": self.text_to_translate, "target_language": self.target_language}

        received_chunks = []
        for chunk in self.agent.stream(input_data):
            received_chunks.append(chunk.content if hasattr(chunk, 'content') else chunk)

        self.assertTrue(len(received_chunks) > 0, "Should have received chunks from the stream.")
        full_response = "".join(received_chunks)
        logger.debug(f"Full streamed response (sync): {full_response}")
        self.assertIn("안녕하세요, 세계!", full_response, "Full response should contain translated text.")
        self.assertIn("Korean", full_response, "Full response should contain target language.")
        mock_stream.assert_called_once_with(input_data)

    async def test_integration_atranslate_real_llm_call(self):
        """Integration test: Calls the real LLM and checks for successful parsing."""
        if not self.agent: self.skipTest("Agent not created in setUp")
        
        logger.info("\n=== RUNNING REAL LLM INTEGRATION TEST (TRANSLATION) ===")
        logger.info(f"Agent Model: {self.agent.effective_llm_provider}/{self.agent.effective_model_name}")
        
        input_text = "hello, how are you?"
        target_language = "Korean"

        print(f"\nIntegration Test Input:")
        print(f"  Text to translate: '{input_text}'")
        print(f"  Target language: {target_language}\n")

        try:
            response = await self.agent.atranslate(input_text, target_language)
            
            print("Integration Test - Raw LLM Response (Successfully Parsed by TranslationAgent):")
            if isinstance(response, TranslationResponse):
                print(response.model_dump_json(indent=2))
            else:
                print(f"Unexpected response type: {type(response)}, content: {str(response)}")

            self.assertIsInstance(response, TranslationResponse, "The response should be a TranslationResponse object.")
            self.assertIsNotNone(response.translated_text, "Translated text should not be None.")
            self.assertGreater(len(response.translated_text), 0, "Translated text should not be empty.")
            self.assertEqual(response.target_language.lower(), target_language.lower(), "Target language should match.")
            self.assertEqual(response.original_text, input_text, "Original text should match input.")

            logger.info("Integration test PASSED: Response successfully parsed into TranslationResponse.")

        except OutputParserException as ope:
            logger.error(f"Integration test FAILED: OutputParserException: {ope}", exc_info=True)
            if hasattr(ope, 'llm_output') and ope.llm_output: print(f"LLM Output that failed parsing:\n{ope.llm_output}")
            self.fail(f"OutputParserException occurred: {ope}")
        except ChainError as ce:
            logger.error(f"Integration test FAILED: ChainError: {ce}", exc_info=True)
            self.fail(f"ChainError occurred: {ce}")
        except Exception as e:
            logger.error(f"Integration test FAILED: Unexpected error: {e}", exc_info=True)
            self.fail(f"An unexpected error occurred: {e}")
        finally:
            logger.info("=== FINISHED REAL LLM INTEGRATION TEST (TRANSLATION) ===\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run TranslationAgent tests with custom LLM settings.")
    parser.add_argument(
        "--llm-provider", 
        type=str, 
        choices=["openai", "anthropic", "ollama", "vllm"], 
        help="Specify the LLM provider."
    )
    parser.add_argument(
        "--llm-model", 
        type=str, 
        help="Specify the LLM model name."
    )
    # 추가적인 인자(예: API 키, URL)를 받고 싶다면 여기에 추가

    args, unknown = parser.parse_known_args()

    # 전역 langchain_settings 객체를 가져옴 (이 시점에서는 YAML 기본값으로 초기화된 상태)
    from app.langgraph.langchain.settings import langchain_settings

    if args.llm_provider or args.llm_model:
        logger.info("Command line arguments provided for LLM settings. Overriding YAML settings.")
        
        override_kwargs = {}
        if args.llm_provider:
            override_kwargs['model_provider'] = args.llm_provider
            logger.info(f"  Overriding model_provider with: {args.llm_provider}")
        
        if args.llm_model:
            override_kwargs['model_name'] = args.llm_model
            logger.info(f"  Overriding model_name with: {args.llm_model}")

        try:
            custom_settings = LangChainSettings(**override_kwargs)
            set_langchain_settings(custom_settings)
            logger.info(f"--- Applied custom test settings from CLI: {custom_settings.model_dump_json(indent=2)} ---")
        except Exception as e:
            logger.error(f"Failed to apply custom settings from CLI: {e}", exc_info=True)
            logger.warning("Proceeding with default YAML settings due to error in applying CLI arguments.")
    else:
        logger.info("No CLI arguments for LLM settings. Using default settings from YAML / Pydantic.")
        # 전역 langchain_settings는 이미 모듈 임포트 시 초기화되었으므로 별도 작업 불필요

    # unittest가 나머지 인자(예: -v)를 처리하도록 unknown을 다시 argv에 추가
    sys.argv = [sys.argv[0]] + unknown 
    unittest.main()
