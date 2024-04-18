import argparse
import csv
import glob
import os
import statistics
import sys

import matplotlib.ticker

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sortedcontainers import SortedSet
from netconfeval.utils import strtobool

model2plot = {
    "gpt-4-turbo": {
        "label": "GPT-4-Turbo",
        "color": "#377eb8",
        "marker": "o"
    },
    "gpt-4": {
        "label": "GPT-4",
        "color": "#e41a1c",
        "marker": "s"
    },
    "gpt-3.5-turbo": {
        "label": "GPT-3.5-Turbo",
        "color": "#4daf4a",
        "marker": ">"
    },
    "gpt-4-combined": {
        "label": "GPT-4 (Combined)",
        "color": "#984ea3",
        "marker": "<"
    },
}


def extract_result(file_path: str, batch_num: int) -> (dict, dict):
    average = {}

    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for res in reader:
            n_req = int(res["batch_size"]) * batch_num
            if n_req not in average:
                average[n_req] = {
                    "batch_size": int(res["batch_size"]),
                    "n_policy_types": int(res["n_policy_types"]),
                    "max_n_requirements": int(res["max_n_requirements"]),
                    "TP": {},
                    "FP": {},
                    "FN": {},
                    "TN": {},
                }

            it = int(res["iteration"])

            if it not in average[n_req]["TP"]:
                average[n_req]["TP"][it] = 0
            if it not in average[n_req]["FN"]:
                average[n_req]["FN"][it] = 0
            if it not in average[n_req]["FP"]:
                average[n_req]["FP"][it] = 0
            if it not in average[n_req]["TN"]:
                average[n_req]["TN"][it] = 0

            if strtobool(res["conflict_exist"]):
                if strtobool(res["conflict_detect"]):
                    average[n_req]["TP"][it] += 1
                else:
                    average[n_req]["FN"][it] += 1
            else:
                if strtobool(res["conflict_detect"]):
                    average[n_req]["FP"][it] += 1
                else:
                    average[n_req]["TN"][it] += 1

    to_plot_accuracy = {"x": [], "y": [], "min_y": [], "max_y": []}
    to_plot_precision = {"x": [], "y": [], "min_y": [], "max_y": []}
    to_plot_recall = {"x": [], "y": [], "min_y": [], "max_y": []}
    to_plot_f1_score = {"x": [], "y": [], "min_y": [], "max_y": []}

    for n_req, avg in average.items():
        it_acc = []
        it_pre = []
        it_rec = []
        it_f1 = []
        for n_run in avg['TP'].keys():
            # print(avg["TP"][n_run], avg["TN"][n_run], avg["FP"][n_run], avg["FN"][n_run])
            accuracy = ((avg["TP"][n_run] + avg["TN"][n_run]) /
                        (avg["TP"][n_run] + avg["TN"][n_run] + avg["FP"][n_run] + avg["FN"][n_run])
                        )
            if (avg["TP"][n_run] + avg["FP"][n_run]) == 0:
                precision = 1
            else:
                precision = avg["TP"][n_run] / (avg["TP"][n_run] + avg["FP"][n_run])

            if (avg["TP"][n_run] + avg["FN"][n_run]) == 0:
                recall = 1
            else:
                recall = avg["TP"][n_run] / (avg["TP"][n_run] + avg["FN"][n_run])

            print(n_req, n_run, precision, recall, accuracy)
            if precision + recall == 0:
                f1_score = 1
            else:
                f1_score = (2 * precision * recall) / (precision + recall)
            it_acc.append(accuracy)
            it_pre.append(precision)
            it_rec.append(recall)
            it_f1.append(f1_score)

        to_plot_accuracy["x"].append(n_req)
        to_plot_accuracy["y"].append(statistics.mean(it_acc))
        to_plot_accuracy["min_y"].append(min(it_acc))
        to_plot_accuracy["max_y"].append(max(it_acc))
        to_plot_precision["x"].append(n_req)
        to_plot_precision["y"].append(statistics.mean(it_pre))
        to_plot_precision["min_y"].append(min(it_pre))
        to_plot_precision["max_y"].append(max(it_pre))
        to_plot_recall["x"].append(n_req)
        to_plot_recall["y"].append(statistics.mean(it_rec))
        to_plot_recall["min_y"].append(min(it_rec))
        to_plot_recall["max_y"].append(max(it_rec))
        to_plot_f1_score["x"].append(n_req)
        to_plot_f1_score["y"].append(statistics.mean(it_f1))
        to_plot_f1_score["min_y"].append(min(it_f1))
        to_plot_f1_score["max_y"].append(max(it_f1))

    return to_plot_accuracy, to_plot_precision, to_plot_recall, to_plot_f1_score


def plot_by_requirements(results_path: str, figures_path: str, requirements: SortedSet) -> None:
    model2result = {}
    requirements_str = "_".join(requirements)

    model_list = model2plot.keys()
    for model_name in model_list:
        results_files_list = glob.glob(
            os.path.join("evaluation", results_path, f"result-{model_name}-{requirements_str}-conflict-*.csv"))

        if len(results_files_list) == 0:
            continue

        if results_files_list:
            results_file = results_files_list.pop()

            if model_name not in model2result:
                model2result[model_name] = {}

            print(model_name)
            model2result[model_name]["accuracy"], model2result[model_name]["precision"], model2result[model_name][
                "recall"], model2result[model_name]["f1_score"] = extract_result(results_file, len(requirements))
        print(model2result[model_name]["accuracy"], model2result[model_name]["precision"],
              model2result[model_name]["recall"], model2result[model_name]["f1_score"])

    base_figures_path = os.path.join("evaluation", figures_path)
    os.makedirs(base_figures_path, exist_ok=True)

    for param in ["accuracy", "precision", "recall", "f1_score"]:
        plt.clf()
        ax = plt.gca()

        for model, results in model2result.items():
            model_params = model2plot[model]
            plt.plot(results[param]["x"], results[param]["y"],
                     marker=model_params["marker"],
                     fillstyle='none',
                     linestyle='--',
                     color=model_params["color"],
                     label=model_params["label"]
                     )

            for idx, x in enumerate(results[param]['x']):
                plt.errorbar(
                    x,
                    results[param]['y'][idx],
                    yerr=[[results[param]['y'][idx] - results[param]['min_y'][idx]],
                          [results[param]['max_y'][idx] - results[param]['y'][idx]]],
                    color=model_params["color"],
                    elinewidth=1, capsize=1
                )

        plt.ylim([0, 1.2])
        plt.yticks(np.arange(0, 1.2, 0.25))
        plt.xscale('log', base=10)
        plt.xticks([3, 9, 33, 99])
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.legend(loc='lower left' if param == "f1_score" else "lower right", labelspacing=0.2, ncol=1, prop={'size': 8})
        plt.xlabel('Batch Size')
        label = "Accuracy" if param == "accuracy" else "Precision" if param == "precision" else "Recall" if param == "recall" else "F1-Score" if param == "f1_score" else "None"
        plt.ylabel(label)
        plt.grid(True)
        plt.savefig(os.path.join(base_figures_path, f"conflict-{requirements_str}-{param}.pdf"),
                    format="pdf", bbox_inches='tight')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, required=False, default="result")
    parser.add_argument('--figures_path', type=str, required=True)

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    plt.figure(figsize=(4, 2))
    mpl.rc('font', size=10)
    mpl.rcParams['hatch.linewidth'] = 0.3
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    plot_by_requirements(args.results_path, args.figures_path, SortedSet({"loadbalancing", "reachability", "waypoint"}))

    # plt.plot(req_bard, accuracy_bard, marker='s', fillstyle='none', linestyle='--', color='gold', label='BARD')


if __name__ == "__main__":
    main(parse_args())
