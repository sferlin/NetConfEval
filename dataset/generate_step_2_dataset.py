import argparse
import json
import os
import re
import sys

from sortedcontainers import SortedDict

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from netconfeval.common.utils import *


def find_test_files(path: str) -> SortedDict:
    test_map = SortedDict()
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".py"):
                match = re.match(r'^(\d+)_', file)
                if match:
                    order = int(match.group(1))
                    test_map[order] = os.path.join(path, file)

    return test_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results_path', type=str, default=os.path.join("..", "datasets")
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        format='[%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    os.makedirs(args.results_path, exist_ok=True)

    code_base_assets_path = os.path.abspath(os.path.join('..', 'assets', 'step_2_code_base'))
    tests_assets_path = os.path.abspath(os.path.join('..', 'assets', 'step_2_tests'))

    with open(os.path.join(args.results_path, "step_2_code_gen.jsonl"), "w") as f:
        for prompts in ["basic", "no_detail"]:
            if prompts == "basic":
                from netconfeval.prompts.step_2_basic import INPUT_OUTPUT_PROMPT, INSTRUCTION_PROMPT, \
                    ADD_REQUIREMENT_REACHABILITY_PROMPT, ADD_REQUIREMENT_WAYPOINT_PROMPT, \
                    ADD_REQUIREMENT_LOADBALANCING_PROMPT
            if prompts == "no_detail":
                from netconfeval.prompts.step_2_no_detail import INPUT_OUTPUT_PROMPT, INSTRUCTION_PROMPT, \
                    ADD_REQUIREMENT_REACHABILITY_PROMPT, ADD_REQUIREMENT_WAYPOINT_PROMPT, \
                    ADD_REQUIREMENT_LOADBALANCING_PROMPT

            for policy in ["shortest_path", "reachability", "waypoint", "loadbalancing"]:
                logging.warning(f"Generating line for policy={policy} and prompts={prompts}...")

                test_cases_path = ["path_tests"]

                code_path = os.path.join(code_base_assets_path, "computing_path_empty.py")
                add_requirement_prompts = ""

                if policy == "reachability":
                    code_path = os.path.join(code_base_assets_path, "computing_path_shortest_path.py")
                    add_requirement_prompts = ADD_REQUIREMENT_REACHABILITY_PROMPT
                    test_cases_path.append("reachability_tests")
                elif policy == "waypoint":
                    code_path = os.path.join(code_base_assets_path, "computing_path_reachability.py")
                    add_requirement_prompts = ADD_REQUIREMENT_WAYPOINT_PROMPT
                    test_cases_path.append("reachability_tests")
                    test_cases_path.append("waypoint_tests")
                elif policy == "loadbalancing":
                    code_path = os.path.join(code_base_assets_path, "computing_path_shortest_path.py")
                    add_requirement_prompts = ADD_REQUIREMENT_LOADBALANCING_PROMPT
                    test_cases_path.append("reachability_tests")
                    test_cases_path.append("waypoint_tests")
                    test_cases_path.append("loadbalancing_tests")

                with open(code_path, "r") as code_fd:
                    base_code = code_fd.read()
                prompt = base_code + "\n" + add_requirement_prompts

                combined_human_prompt = f"{INPUT_OUTPUT_PROMPT}\n{INSTRUCTION_PROMPT}\n{prompt}"
                test_cases_files = SortedDict()
                for tests_path in test_cases_path:
                    files = find_test_files(os.path.join(tests_assets_path, tests_path))
                    test_cases_files.update(files)

                for test_num, test_loc in test_cases_files.items():
                    with open(test_loc, 'r') as test_fd:
                        test_case = test_fd.read()
                    test_cases_files[test_num] = test_case

                result_row = {
                    'prompts': prompts,
                    'policy': policy,
                    'input': combined_human_prompt.replace('{{', '{').replace('}}', '}'),
                    'tests': json.dumps(test_cases_files)
                }

                f.write(json.dumps(result_row) + "\n")
                f.flush()


if __name__ == "__main__":
    main(parse_args())
