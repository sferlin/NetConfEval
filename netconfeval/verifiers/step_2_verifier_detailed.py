import logging
import os
import re
import sys
import tempfile
from collections import OrderedDict
from subprocess import Popen, PIPE

from sortedcontainers import SortedDict

from netconfeval.foundation.verifier.verifier import Verifier


class Step2VerifierDetailed(Verifier):
    __slots__ = ['_tests']

    def __init__(self, default_test_path: str) -> None:
        self._tests: OrderedDict[str, SortedDict] = OrderedDict()
        self._tests[default_test_path] = self._find_test_files(default_test_path)

    def verify(self, response: str, data: dict | None = None) -> (bool, str):
        logging.warning(f"Called Step2VerifierDetailed with response={response}, data={data}")

        response = self._sanitize_output(response)
        path = None if 'test_path' not in data['metadata'] else data['metadata']['test_path']
        if path is not None and path not in self._tests:
            self._tests[path] = self._find_test_files(path)

        logging.debug(f"Running tests in {path}...")
        for order, test_file_path in self._tests[path].items():
            logging.debug(f"Verifying {test_file_path} now...")
            success, output = self._tester(response, test_file_path)
            if not success:
                return False, output

        return True, None

    def _tester(self, code: str, test_loc: str) -> (bool, str):
        logging.debug(test_loc)
        with open(test_loc, 'r') as f:
            test_case = f.read()

        test_case = test_case.replace("# ~function_code~", code)

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_file_path = os.path.join(tmp_dir, 'test_code.py')
            with open(temp_file_path, 'w') as temp_file:
                temp_file.writelines(test_case)

            process = Popen(
                f"{sys.executable} -m pytest --lf --tb=short {temp_file_path} -vv", stdout=PIPE, shell=True
            )
            output, _ = process.communicate()
            process.terminate()

        output = output.decode('utf-8')
        logging.warning(f"Test output {output}")
        logging.debug(f"Test return code {process.returncode}")

        return self.give_feedback(output, process.returncode)

    @staticmethod
    def give_feedback(output: str, return_code: int) -> (bool, str):
        feedback = output
        if return_code != 0:
            if "AssertionError" not in output:
                failure_pattern = re.compile(
                    r"=================================== FAILURES ===================================\n(.*)",
                    re.DOTALL
                )
                test_feedback = failure_pattern.search(output)

                if test_feedback:
                    feedback = test_feedback.group(1).strip()

                logging.debug(feedback)

                return False, f"In the following you will find the pytest output:\n{feedback}"
            else:
                message_pattern = re.compile(r"E\s+AssertionError:(.*)\nE\s+assert", re.DOTALL)
                message = message_pattern.search(output)

                if message:
                    feedback = re.sub(r'E\s+', '', message.group(1).strip())

                logging.debug(feedback)

            return False, feedback
        else:
            return True, None

    @staticmethod
    def _find_test_files(path: str) -> SortedDict:
        test_map = SortedDict()
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".py"):
                    match = re.match(r'^(\d+)_', file)
                    if match:
                        order = int(match.group(1))
                        test_map[order] = os.path.join(path, file)

        return test_map

    @staticmethod
    def _sanitize_output(text: str):
        if "```python" in text:
            _, after = text.split("```python")
            return after.split("```")[0]
        else:
            return text
