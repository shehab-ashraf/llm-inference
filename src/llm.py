"""User-facing LLM interface."""
from src.engine.llm_engine import GenerationResult, LLMEngine
from src.sampling_params import SamplingParams


class LLM(LLMEngine):
    pass


__all__ = ["LLM", "GenerationResult", "SamplingParams"]
