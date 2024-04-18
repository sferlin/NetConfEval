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
    "gpt-4-turbo--": {
        "label": "gpt-4-turbo",
        "color": "#377eb8",
        "marker": "s"
    },
    "gpt-4-turbo-function--": {
        "label": "gpt-4-turbo-function",
        "color": "#984ea3",
        "marker": "^"
    },
    "gpt-4-turbo-function-ad_hoc": {
        "label": "gpt-4-turbo-function-ad_hoc",
        "color": "#a65628",
        "marker": "o"
    },
}


def extract_result(file_path: str, model_name: str) -> (dict, dict):
    average = {}

    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for res in reader:
            n_req = int(res["batch_size"])
            # for a figure
            # if n_req * int(res["n_policy_types"]) == 25:
            #     continue

            if n_req not in average:
                average[n_req] = {
                    "batch_size": int(res["batch_size"]),
                    "n_policy_types": int(1),
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

            cost = float(res["total_cost"]) / int(n_req)

            if model_name == "gpt-4-turbo-function":
                cost = (float(res["prompt_tokens"]) * 0.01 + float(res["completion_tokens"]) * 0.03) / (
                        int(n_req) * int(1)) / 1000
            elif model_name == "gpt-3.5-finetuned":
                cost = (float(res["prompt_tokens"]) * 0.012 + float(res["completion_tokens"]) * 0.016) / (
                        int(n_req) * int(1)) / 1000

            if cost == 0:
                continue
            average[n_req]["cost"][it].append(cost)

    to_plot_accuracy = {"x": [], "y": [], "min_y": [], "max_y": []}
    to_plot_cost = {"x": [], "y": [], "min_y": [], "max_y": []}

    for n_req, avg in average.items():
        # print(avg["accuracy"])
        # exit()
        it_acc = []
        for accuracy in avg["accuracy"].values():
            it_acc.append(
                sum([x * avg["batch_size"] for x in accuracy]) / (avg["batch_size"] / avg["n_policy_types"])
            )

        to_plot_accuracy["x"].append(n_req)
        to_plot_accuracy["y"].append(statistics.mean(it_acc) if len(it_acc) >= 1 else it_acc[0])
        to_plot_accuracy["min_y"].append(min(it_acc))
        to_plot_accuracy["max_y"].append(max(it_acc))

        it_cost = []
        for cost in avg["cost"].values():
            it_cost.append(
                statistics.mean(cost) if len(cost) >= 1 else 0
            )

        to_plot_cost["x"].append(n_req * int(1))
        to_plot_cost["y"].append(
            statistics.mean(it_cost)
        )
        to_plot_cost["min_y"].append(min(it_cost))
        to_plot_cost["max_y"].append(max(it_cost))

    return to_plot_accuracy, to_plot_cost


def plot_by_requirements(results_path: str, figures_path: str, requirements_1: SortedSet, requirements_2: SortedSet,
                         requirements_3: SortedSet) -> None:
    model2result = {}

    reqs = [requirements_1, requirements_2, requirements_3]

    plt.figure(figsize=(4, 2))
    ax = plt.gca()
    # for requirements in reqs:
    #     requirements_str = "_".join(requirements)
    #     print(requirements_str)

    model_name = "gpt-4-turbo"

    results_files_list = glob.glob(os.path.join("evaluation", results_path, f"result-{model_name}-*.csv"))
    count = 1
    while results_files_list:
        results_file = results_files_list.pop()

        # print(results_file)

        file_name = str(results_file)

        if "gpt-4-turbo--" in results_file:
            file_name = "gpt-4-turbo--"
        elif "gpt-4-turbo-function--" in results_file:
            file_name = "gpt-4-turbo-function--"
        elif "gpt-4-turbo-function-ad_hoc" in results_file:
            file_name = "gpt-4-turbo-function-ad_hoc"

        count += 1

        if file_name not in model2result:
            model2result[file_name] = {}

        model2result[file_name]["accuracy"], model2result[file_name]["cost"] = extract_result(
            results_file, model_name)

    # print(model2result)
    # exit()

    base_figures_path = os.path.join("evaluation", figures_path)
    os.makedirs(base_figures_path, exist_ok=True)

    print("Model result: \n", model2result)

    for file_name, results in model2result.items():
        model_params = model2plot[file_name]
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
    plt.xticks(model2result[list(model2plot.keys())[0]]["accuracy"]["x"])
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.legend(loc="lower left", labelspacing=0.1, ncol=1)
    plt.xlabel('Batch Size')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(base_figures_path, f"accuracy-function-call.pdf"),
                format="pdf", bbox_inches='tight')

    # Cost
    plt.clf()
    mpl.rc('font', size=10)
    plt.figure(figsize=(4, 2.2))
    ax = plt.gca()

    for requirements_str, results in model2result.items():
        if any([x > 0 for x in results["cost"]["y"]]):
            model_params = model2plot[requirements_str]
            print(results["cost"]["y"])
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
    print(model2result[list(model2plot.keys())[0]]["accuracy"]["x"])
    plt.xticks(model2result[list(model2plot.keys())[0]]["accuracy"]["x"])
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.legend(loc="lower left", labelspacing=0.1, ncol=1)
    plt.xlabel('Batch Size')
    plt.ylabel('Cost [$]')
    plt.yscale('log', base=10)
    plt.ylim(0.0003, 0.005)
    plt.yticks(
        [0.0003, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005],
        ['$3\\times10^{-4}$', '$4\\times10^{-4}$', '$10^{-3}$', '$2\\times10^{-3}$', '$3\\times10^{-3}$', None, None]
    )
    plt.grid(True)
    plt.savefig(os.path.join(base_figures_path, f"cost-function-call.pdf"),
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

    plot_by_requirements(args.results_path, args.figures_path, SortedSet({"reachability"}),
                         SortedSet({"reachability", "waypoint"}),
                         SortedSet({"loadbalancing", "reachability", "waypoint"}))
    # plot_by_requirements(args.results_path, args.figures_path, SortedSet({"reachability", "waypoint"}))
    # plot_by_requirements(args.results_path, args.figures_path, SortedSet({"loadbalancing", "reachability", "waypoint"}))

    # plt.plot(req_bard, accuracy_bard, marker='s', fillstyle='none', linestyle='--', color='gold', label='BARD')


if __name__ == "__main__":
    main(parse_args())
