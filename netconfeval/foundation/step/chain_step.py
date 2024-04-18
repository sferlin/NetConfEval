import json
import logging
from typing import Callable

from langchain.chains.llm import LLMChain

from ..verifier.verifier import Verifier

DEFAULT_FEEDBACK_RETRIES = 10


class ChainStep:
    __slots__ = [
        '_llm_chain', '_verifier', '_feedback_prompt', '_feedback_retries', '_input_formatter', '_output_formatter',
        'failure_number', 'format_error', 'syntax_error', 'test_error'
    ]

    def __init__(self, llm_chain: LLMChain, input_formatter: Callable, output_formatter: Callable,
                 verifier: Verifier | None = None, feedback_prompt: str | None = None,
                 feedback_retries: int | None = None) -> None:
        self._llm_chain: LLMChain = llm_chain
        self._verifier: Verifier | None = verifier
        self._feedback_prompt: str | None = feedback_prompt
        self._feedback_retries: int = feedback_retries if feedback_retries else DEFAULT_FEEDBACK_RETRIES
        self._input_formatter: Callable = input_formatter
        self._output_formatter: Callable = output_formatter

        self.failure_number = 0
        self.format_error = 0
        self.syntax_error = 0
        self.test_error = 0

    def process(self, data: any) -> (bool, any):
        self.failure_number = 0
        self.format_error = 0
        self.syntax_error = 0
        self.test_error = 0

        success = False

        input_data = self._input_formatter(data)
        logging.debug(f"input_data={input_data}")

        result = None
        while not success and self.failure_number < self._feedback_retries:
            response = self._llm_chain.invoke(input_data)
            try:
                output = self.output_parser(response)
                logging.debug(output)
            except json.JSONDecodeError:
                self._llm_chain.memory.clear()
                logging.error(f"Output JSON format error: {response}")
                self.failure_number += 1
                self.format_error += 1
                continue
            try:
                if output["status"].lower() == "error":
                    return False, output

                result = output["result"]
            except KeyError:
                self._llm_chain.memory.clear()
                logging.error(f"Missing \"result\" JSON key: {response}")
                self.failure_number += 1
                self.format_error += 1
                continue

            # Nothing to verify, exit the loop
            if self._verifier is None:
                break

            success, errors = self._verifier.verify(result, data)
            if not success:
                self.failure_number += 1

                # No feedback, just retry
                if self.failure_number >= self._feedback_retries:
                    raise Exception("Too many failures.")

                if self._feedback_prompt is not None:
                    feedback = self.prepare_feedback_prompts(errors)

                    logging.warning(feedback)
                    if "Code generation wrongly" in feedback:
                        self.test_error += 1
                    else:
                        self.syntax_error += 1

                    input_data = {"input": feedback}

        logging.warning(result)

        return True, self._output_formatter(result)

    def prepare_feedback_prompts(self, feedback: str) -> str:
        return self._feedback_prompt.format(feedback=feedback)

    def output_parser(self, text: dict) -> dict:
        return json.loads(text["text"])
