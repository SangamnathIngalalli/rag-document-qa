import os
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic
from deepeval.models import DeepEvalBaseLLM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class ClaudeJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "claude-sonnet-5"):
        self.model_name = model_name
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema=None) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        )

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self.model_name


judge = ClaudeJudge()