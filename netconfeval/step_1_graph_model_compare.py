import argparse
import csv
import glob
import os.path
import statistics

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sortedcontainers import SortedSet

model2plot = {
    "gpt-4-turbo": {
        "label": "GPT-4-Turbo",
        "color": "#377eb8",
        "marker": "o"
    },
    "gpt-3.5-finetuned": {
        "label": "GPT-3.5-FT",
        "color": "#a65628",
        "marker": "s"
    },
    "gpt-3.5-turbo": {
        "label": "GPT-3.5-Turbo",
        "color": "#984ea3",
        "marker": ">"
    },
    "gpt-4-turbo-function": {
        "label": "gpt-4-turbo-function",
        "color": "#e41a1c",
        "marker": "1"
    },
    # "gpt-4": {
    #     "label": "GPT-4",
    #     "color": "#ffff33",
    #     "marker": "+"
    # },
    "codellama-13b-instruct": {
        "label": "CL-13B-Instruct",
        "color": "#ff7f00",
        "marker": "<"
    },
    "codellama-7b-instruct-finetuned": {
        "label": "CL-7B-Instruct-FT (QLoRA)",
        "color": "#4daf4a",
        "marker": ">"
    },
    "codellama-7b-instruct": {
        "label": "CL-7B-Instruct",
        "color": "#f781bf",
        "marker": "^"
    },
}


def extract_result(file_path: str, model_name: str) -> (dict, dict):
    average = {}

    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for res in reader:
            n_req = int(res["batch_size"])
            # for a figure
            if n_req * int(res["n_policy_types"]) == 25:
                continue

            if n_req not in average:
                average[n_req] = {
                    "batch_size": int(res["batch_size"]),
                    "n_policy_types": int(res["n_policy_types"]),
                    "max_n_requirements": int(res["max_n_requirements"]),
                    "accuracy": {},
                    "cost": {},
                }

            it = int(res["iteration"])
            if it not in average[n_req]["accuracy"]:
                average[n_req]["accuracy"][it] = []
            if it not in average[n_req]["cost"]:
                average[n_req]["cost"][it] = []

            average[n_req]["accuracy"][it].append(float(res["accuracy"]))

            cost = float(res["total_cost"]) / (int(n_req) * int(res["n_policy_types"]))

            if model_name == "gpt-4-turbo-function":
                cost = (float(res["prompt_tokens"]) * 0.01 + float(res["completion_tokens"]) * 0.03) / (
                        int(n_req) * int(res["n_policy_types"])) / 1000
            elif model_name == "gpt-3.5-finetuned":
                cost = (float(res["prompt_tokens"]) * 0.012 + float(res["completion_tokens"]) * 0.016) / (
                        int(n_req) * int(res["n_policy_types"])) / 1000

            if cost == 0:
                continue
            average[n_req]["cost"][it].append(cost)

    to_plot_accuracy = {"x": [], "y": [], "min_y": [], "max_y": []}
    to_plot_cost = {"x": [], "y": [], "min_y": [], "max_y": []}
    for n_req, avg in average.items():
        it_acc = []
        for accuracy in avg["accuracy"].values():
            it_acc.append(
                sum([x * avg["batch_size"] for x in accuracy]) / (avg["max_n_requirements"] / avg["n_policy_types"])
            )

        to_plot_accuracy["x"].append(n_req * int(res["n_policy_types"]))
        to_plot_accuracy["y"].append(statistics.mean(it_acc) if len(it_acc) >= 1 else it_acc[0])
        to_plot_accuracy["min_y"].append(min(it_acc))
        to_plot_accuracy["max_y"].append(max(it_acc))

        it_cost = []
        for cost in avg["cost"].values():
            it_cost.append(
                statistics.mean([x for x in cost if x > 0]) if len(cost) > 0 else 0
            )

        it_cost = [x for x in it_cost if x > 0]

        to_plot_cost["x"].append(n_req * int(res["n_policy_types"]))
        to_plot_cost["y"].append(
            statistics.mean(it_cost) if len(it_cost) > 0 else 0
        )
        to_plot_cost["min_y"].append(min(it_cost) if len(it_cost) > 0 else 0)
        to_plot_cost["max_y"].append(max(it_cost) if len(it_cost) > 0 else 0)

    return to_plot_accuracy, to_plot_cost


def plot_by_requirements(results_path: str, figures_path: str, include: str, requirements: SortedSet) -> None:
    model2result = {}
    requirements_str = "_".join(requirements)

    for model_name in model2plot.keys():
        if include not in model_name or 'function' in model_name:
            continue

        results_files_list = glob.glob(os.path.join("evaluation", results_path, f"result-{model_name}-{requirements_str}-*.csv"))
        if results_files_list:
            results_file = results_files_list.pop()

            if model_name not in model2result:
                model2result[model_name] = {}

            model2result[model_name]["accuracy"], model2result[model_name]["cost"] = extract_result(results_file,
                                                                                                    model_name)

    base_figures_path = os.path.join("evaluation", figures_path)
    os.makedirs(base_figures_path, exist_ok=True)

    # Accuracy
    plt.figure(figsize=(4, 2))
    ax = plt.gca()

    for model, results in model2result.items():
        model_params = model2plot[model]
        print(results["accuracy"]["y"])
        plt.plot(results["accuracy"]["x"], results["accuracy"]["y"],
                 marker=model_params["marker"],
                 fillstyle='none',
                 linestyle='--',
                 color=model_params["color"],
                 label=model_params["label"]
                 )

        for idx, x in enumerate(results["accuracy"]['x']):
            plt.errorbar(
                x,
                results["accuracy"]['y'][idx],
                yerr=[[results["accuracy"]['y'][idx] - results["accuracy"]['min_y'][idx]],
                      [results["accuracy"]['max_y'][idx] - results["accuracy"]['y'][idx]]],
                color=model_params["color"],
                elinewidth=1, capsize=1
            )

    plt.ylim([-0.1, 1.2])
    plt.yticks(np.arange(0, 1.2, 0.25))
    plt.xscale('log', base=10)
    x_ticks = list(model2result.values())[0]["accuracy"]["x"]
    plt.xticks(x_ticks)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.legend(loc="lower left", labelspacing=0.1, ncol=1, prop={'size': 9})
    plt.xlabel('Batch Size')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(base_figures_path, f"accuracy-{requirements_str}.pdf"),
                format="pdf", bbox_inches='tight')

    # Cost
    plt.figure(figsize=(4, 2))
    ax = plt.gca()
    for model, results in model2result.items():
        if any([x > 0 for x in results["cost"]["y"]]):
            model_params = model2plot[model]
            results["cost"]["y"] = list(filter(lambda x: x > 0, results["cost"]["y"]))
            results["cost"]["x"] = results["cost"]["x"][0:len(results["cost"]["y"])]
            plt.plot(results["cost"]["x"], results["cost"]["y"],
                     marker=model_params["marker"],
                     fillstyle='none',
                     linestyle='--',
                     color=model_params["color"],
                     label=model_params["label"]
                     )

            for idx, x in enumerate(results["cost"]['x']):
                plt.errorbar(
                    x,
                    results["cost"]['y'][idx],
                    yerr=[[results["cost"]['y'][idx] - results["cost"]['min_y'][idx]],
                          [results["cost"]['max_y'][idx] - results["cost"]['y'][idx]]],
                    color=model_params["color"],
                    elinewidth=1, capsize=1
                )

    plt.xscale('log', base=10)
    x_ticks = list(model2result.values())[0]["accuracy"]["x"]
    plt.xticks(x_ticks)
    plt.ylim(0.00001, 0.02)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.legend(loc='lower left', labelspacing=0.1, ncol=1, prop={'size': 9})
    plt.xlabel('Batch Size')
    plt.ylabel('Cost [$]')
    plt.yscale('log', base=10)
    plt.grid(True)
    plt.savefig(os.path.join(base_figures_path, f"cost-{requirements_str}.pdf"),
                format="pdf", bbox_inches='tight')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, required=False, default="result")
    parser.add_argument('--figures_path', type=str, required=True)
    parser.add_argument('--include', type=str, choices=['gpt', 'codellama'])

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    matplotlib.rc('font', size=10)
    matplotlib.rcParams['hatch.linewidth'] = 0.3
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    plot_by_requirements(args.results_path, args.figures_path, args.include, SortedSet({"reachability"}))
    plot_by_requirements(args.results_path, args.figures_path, args.include, SortedSet({"reachability", "waypoint"}))
    plot_by_requirements(args.results_path, args.figures_path, args.include,
                         SortedSet({"loadbalancing", "reachability", "waypoint"}))


if __name__ == "__main__":
    main(parse_args())
