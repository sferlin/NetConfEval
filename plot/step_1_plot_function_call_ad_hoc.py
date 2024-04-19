import argparse
import csv
import glob
import os.path
import statistics

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sortedcontainers import SortedSet

model2plot = {
    "gpt-3.5-1106-adhoc": {
        "label": "GPT-3.5-Turbo",
        "color": "#984ea3",
        "marker": ">"
    },
    "gpt-4-1106-native": {
        "label": "GPT-4-Turbo (Native)",
        "color": "#e41a1c",
        "marker": "s"
    },
    "gpt-4-turbo-adhoc": {
        "label": "GPT-4-Turbo (Ad-hoc)",
        "color": "#377eb8",
        "marker": "o"
    },
    "codellama-7b-instruct-adhoc": {
        "label": "CL-7B-Instruct",
        "color": "#f781bf",
        "marker": "^"
    },
}


def extract_result(file_path: str, model_name: str) -> dict:
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
                }

            it = int(res["iteration"])
            if it not in average[n_req]["accuracy"]:
                average[n_req]["accuracy"][it] = []

            average[n_req]["accuracy"][it].append(float(res["accuracy"]))

    to_plot_accuracy = {"x": [], "y": [], "min_y": [], "max_y": []}
    for n_req, avg in average.items():
        it_acc = []
        for accuracy in avg["accuracy"].values():
            print(n_req, model_name, accuracy)

            it_acc.append(
                sum([x * avg["batch_size"] for x in accuracy]) / (avg["max_n_requirements"] / avg["n_policy_types"])
            )

            print(n_req, model_name, "it_acc", it_acc)

        to_plot_accuracy["x"].append(n_req * int(res["n_policy_types"]))
        to_plot_accuracy["y"].append(statistics.mean(it_acc) if len(it_acc) >= 1 else it_acc[0])
        to_plot_accuracy["min_y"].append(min(it_acc))
        to_plot_accuracy["max_y"].append(max(it_acc))

    return to_plot_accuracy


def plot_by_requirements(results_path: str, figures_path: str, requirements: SortedSet) -> None:
    model2result = {}
    requirements_str = "_".join(requirements)

    for model_name in model2plot.keys():
        results_files_list = glob.glob(
            os.path.join("../", results_path, f"result-{model_name}-{requirements_str}-*.csv")
        )
        if results_files_list:
            results_file = results_files_list.pop()

            if model_name not in model2result:
                model2result[model_name] = {}

            model2result[model_name]["accuracy"] = extract_result(results_file, model_name)

    base_figures_path = os.path.join(".", figures_path)
    os.makedirs(base_figures_path, exist_ok=True)

    # Accuracy
    plt.figure(figsize=(4, 2))
    ax = plt.gca()

    for model, results in model2result.items():
        model_params = model2plot[model]
        print(results["accuracy"]['x'], model_params["label"], results["accuracy"]["y"])
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
    plt.savefig(os.path.join(base_figures_path, f"accuracy-ad-hoc-{requirements_str}.pdf"),
                format="pdf", bbox_inches='tight')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, required=False, default="result")
    parser.add_argument('--figures_path', type=str, required=True)

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    matplotlib.rc('font', size=10)
    matplotlib.rcParams['hatch.linewidth'] = 0.3
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    # plot_by_requirements(args.results_path, args.figures_path, SortedSet({"reachability"}))
    # plot_by_requirements(args.results_path, args.figures_path, SortedSet({"reachability", "waypoint"}))
    plot_by_requirements(args.results_path, args.figures_path, SortedSet({"loadbalancing", "reachability", "waypoint"}))


if __name__ == "__main__":
    main(parse_args())