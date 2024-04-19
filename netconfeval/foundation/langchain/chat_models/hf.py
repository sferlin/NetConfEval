import logging
from typing import Any, List, Optional

from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, AIMessage, ChatResult, ChatGeneration, HumanMessage, SystemMessage
from torch import bfloat16
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)

logger = logging.getLogger(__name__)


def _parse_message(role: str, text: str) -> str:
    return f"{role}: {text}"


def _parse_chat_history(history: List[BaseMessage]) -> list[str]:
    """Parse a sequence of messages into history.

    Returns:
        A tuple of a list of parsed messages and an instruction message for the model.
    """
    chat_history = []
    for message in history:
        if isinstance(message, HumanMessage):
            chat_history.append(_parse_message("User", message.content))
        if isinstance(message, AIMessage):
            chat_history.append(_parse_message("AI", message.content))
        if isinstance(message, SystemMessage):
            chat_history.append(_parse_message("System", message.content))

    return chat_history


class ChatHF(BaseChatModel):
    model_name: str
    max_length: int
    temperature: float | None
    use_quantization: bool
    text_pipeline: Any
    prompt_func: Any

    # Need to set the LD_LIBRARY_PATH to use quantization (BitsAndBytesConfig)
    # You should find the right path for the cuda library via find
    # For example: find /opt/cuda/ -name "libcusparse.so"
    # and then add it to the path For example: export LD_LIBRARY_PATH="/opt/cuda/11.8.0/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH"
    # And then run `python -m bitsandbytes`

    #TODO: add parameter, plot by model name

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model_name = kwargs.get('model_name', None)
        self.max_length = kwargs.get('max_length', 4096)
        self.temperature = kwargs.get('temperature', 0.01)
        self.use_quantization = kwargs.get('use_quantization', False)
        self.prompt_func = kwargs.get('prompt_func', None)
        self.text_pipeline = self._initialize_text_pipeline()

    @property
    def _llm_type(self) -> str:
        return self.model_name

    def _initialize_text_pipeline(self):
        additional_model_kwargs = {}
        if self.use_quantization:
            additional_model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=bfloat16
            )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map='auto',
            **additional_model_kwargs
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        # Can be tuned using online replicate demo for a model
        # For example, check: https://replicate.com/meta/llama-2-70b-chat
        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
            device_map='auto',
            do_sample=True,
            top_p=1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_length=self.max_length,
            temperature=self.temperature,
            repetition_penalty=1,
        )

        return text_pipeline

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> ChatResult:
        chat_history = _parse_chat_history(messages)
        input_text = self.prompt_func(chat_history)
        generation = self.text_pipeline(input_text)
        response_text = generation[0]['generated_text'][len(input_text):]
        message = AIMessage(content=response_text)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError(
            """HF doesn't support async requests at the moment."""
        )
