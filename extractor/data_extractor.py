import csv
import glob
import json
import os
from collections import OrderedDict

from sortedcontainers import SortedSet


def step_1_translation_extract(results_path: str, requirements: set | SortedSet, model_name: str, metric: str) -> dict:
    if metric not in ['accuracy', 'cost']:
        raise Exception(f"Unsupported metric `{metric}`!")

    requirements_str = "_".join(SortedSet(requirements))

    results_files_list = glob.glob(
        os.path.join(results_path, f"result-{model_name}-{requirements_str}-*.csv")
    )
    if len(results_files_list) == 0:
        return {}

    results_file = results_files_list.pop()
    return _step_1_translation_extract_by_metric(results_file, metric)


def step_1_function_call_extract(
        results_path: str, requirements: set | SortedSet, function_call_type: str, model_name: str, metric: str
) -> dict:
    if metric not in ['accuracy', 'cost']:
        raise Exception(f"Unsupported metric `{metric}`!")

    requirements_str = "_".join(SortedSet(requirements))

    results_files_list = glob.glob(
        os.path.join(results_path, f"result-{model_name}-{function_call_type}-{requirements_str}-function-*.csv")
    )
    if len(results_files_list) == 0:
        return {}

    results_file = results_files_list.pop()
    return _step_1_translation_extract_by_metric(results_file, metric)


def step_1_conflict_detection_extract(
        results_path: str, requirements: set | SortedSet, model_name: str, metric: str
) -> dict:
    if metric not in ['accuracy', 'recall', 'f1_score']:
        raise Exception(f"Unsupported metric `{metric}`!")

    requirements_str = "_".join(SortedSet(requirements))

    results_files_list = glob.glob(
        os.path.join(results_path, f"result-{model_name}-{requirements_str}-conflict-*.csv")
    )

    if len(results_files_list) == 0:
        return {}

    results_file = results_files_list.pop()
    return _step_1_conflict_detection_extract_by_metric(results_file, metric)


def step_1_conflict_distance_extract(results_path: str, requirements: set | SortedSet, model_name: str) -> list:
    requirements_str = "_".join(SortedSet(requirements))

    results_files_list = glob.glob(
        os.path.join(results_path, f"result-{model_name}-{requirements_str}-conflict_distance-*.csv")
    )

    data = [[0] * 34] * 34

    if len(results_files_list) == 0:
        return data

    results_file = results_files_list.pop()

    with open(results_file, "r") as file:
        reader = csv.DictReader(file)
        for res in reader:
            if res["conflict_detect"] == "True":
                data[int(res["index_start"])][int(res["index_end"]) - int(res["index_start"])] += 1

    return data


def step_2_code_gen_extract(results_path: str, prompt: str, feedback: str, model_name: str) -> OrderedDict:
    results_files_list = glob.glob(
        os.path.join(results_path, f"result-{model_name}-*-{prompt}-{feedback}-*.csv")
    )
    if len(results_files_list) == 0:
        return OrderedDict()

    results_file = results_files_list.pop()

    data = OrderedDict()
    with open(results_file, 'r') as file:
        reader = csv.DictReader(file)
        for res in reader:
            if res["policy"]:
                policy = res["policy"]
                if policy not in data:
                    data[policy] = {}
                    data[policy]["time"] = []
                    data[policy]["feedback_num"] = []
                    data[policy]["total_cost"] = []

                data[policy]["time"].append(float(res["time"]))
                data[policy]["feedback_num"].append(float(res["feedback_num"]))
                data[policy]["total_cost"].append(float(res["total_cost"]) if "total_cost" in res else 0)

    return data


def step_3_low_level_extract(results_path: str, model_name: str, mode: str, rag_size: int | None = None) -> dict:
    rag_lbl = ""
    if mode == "rag":
        rag_lbl = f"_{rag_size}"
    elif rag_size is not None:
        raise Exception("You must specify `rag_size` only when mode is `rag`!")

    results_files_list = glob.glob(
        os.path.join(results_path, f"result-{model_name}-{mode}{rag_lbl}-*.csv")
    )
    if len(results_files_list) == 0:
        return {}

    results_file = results_files_list.pop()

    data = {}
    with open(results_file, 'r') as file:
        reader = csv.DictReader(file)
        for res in reader:
            scenario_name = res["scenario_name"]
            if scenario_name not in data:
                data[scenario_name] = {}

            it = int(res["iteration"])
            data[scenario_name][it] = json.loads(res["result"])

    return data


def _step_1_translation_extract_by_metric(file_path: str, metric: str) -> dict:
    data = {}

    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for res in reader:
            n_req = int(res["batch_size"])

            if n_req not in data:
                data[n_req] = {
                    "batch_size": int(res["batch_size"]),
                    "n_policy_types": int(res["n_policy_types"]),
                    "max_n_requirements": int(res["max_n_requirements"]),
                    "data": {},
                }

            it = int(res["iteration"])
            if it not in data[n_req]["data"]:
                data[n_req]["data"][it] = []

            if metric == "accuracy":
                data[n_req]["data"][it].append(float(res["accuracy"]))
            elif metric == "cost":
                cost = float(res["total_cost"]) / (int(n_req) * int(res["n_policy_types"]))
                if cost == 0:
                    continue

                data[n_req]["data"][it].append(cost)

    return data


def _step_1_conflict_detection_extract_by_metric(file_path: str, metric: str) -> dict:
    data = {}

    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for res in reader:
            n_req = int(res["batch_size"])

            if n_req not in data:
                data[n_req] = {
                    "batch_size": int(res["batch_size"]),
                    "n_policy_types": int(res["n_policy_types"]),
                    "max_n_requirements": int(res["max_n_requirements"]),
                    "TP": {},
                    "FP": {},
                    "FN": {},
                    "TN": {},
                }

            it = int(res["iteration"])

            if it not in data[n_req]["TP"]:
                data[n_req]["TP"][it] = 0
            if it not in data[n_req]["FN"]:
                data[n_req]["FN"][it] = 0
            if it not in data[n_req]["FP"]:
                data[n_req]["FP"][it] = 0
            if it not in data[n_req]["TN"]:
                data[n_req]["TN"][it] = 0

            if res["conflict_exist"] == "True":
                if res["conflict_detect"] == "True":
                    data[n_req]["TP"][it] += 1
                else:
                    data[n_req]["FN"][it] += 1
            else:
                if res["conflict_detect"] == "True":
                    data[n_req]["FP"][it] += 1
                else:
                    data[n_req]["TN"][it] += 1

    for n_req, res in data.items():
        res["data"] = []

        for n_run in res['TP'].keys():
            accuracy = ((res["TP"][n_run] + res["TN"][n_run]) /
                        (res["TP"][n_run] + res["TN"][n_run] + res["FP"][n_run] + res["FN"][n_run])
                        )

            if (res["TP"][n_run] + res["FP"][n_run]) == 0:
                precision = 1
            else:
                precision = res["TP"][n_run] / (res["TP"][n_run] + res["FP"][n_run])

            if (res["TP"][n_run] + res["FN"][n_run]) == 0:
                recall = 1
            else:
                recall = res["TP"][n_run] / (res["TP"][n_run] + res["FN"][n_run])

            if precision + recall == 0:
                f1_score = 1
            else:
                f1_score = (2 * precision * recall) / (precision + recall)

            if metric == "accuracy":
                res["data"].append(accuracy)
            elif metric == "recall":
                res["data"].append(recall)
            elif metric == "f1_score":
                res["data"].append(f1_score)

    return data
