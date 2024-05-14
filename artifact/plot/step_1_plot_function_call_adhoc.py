import argparse
import os.path
import statistics
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sortedcontainers import SortedSet

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from extractor.data_extractor import step_1_function_call_extract

model2plot = {
    "gpt-3.5-1106_adhoc": {
        "label": "GPT-3.5-Turbo",
        "color": "#984ea3",
        "marker": ">"
    },
    "gpt-4-1106_native": {
        "label": "GPT-4-Turbo (Native)",
        "color": "#e41a1c",
        "marker": "s"
    },
    "gpt-4-1106_adhoc": {
        "label": "GPT-4-Turbo (Ad-hoc)",
        "color": "#377eb8",
        "marker": "o"
    },
    "codellama-7b-instruct_adhoc": {
        "label": "CL-7B-Instruct",
        "color": "#f781bf",
        "marker": "^"
    },
}


def _plot(results_path: str, figures_path: str, requirements: SortedSet) -> None:
    requirements_str = "_".join(requirements)

    results = {}
    for model_name in model2plot.keys():
        name, fn_call_type = model_name.split("_")
        results[model_name] = step_1_function_call_extract(
            results_path, requirements, fn_call_type, name, "accuracy"
        )
    plot_accuracy = False

    plt.clf()
    ax = plt.gca()
    x_ticks = None
    for model, results in results.items():
        to_plot = {"x": [], "y": [], "min_y": [], "max_y": []}
        for n_req, res in results.items():
            it_acc = []
            for accuracy in res["data"].values():
                it_acc.append(
                    sum([x * res["batch_size"] for x in accuracy]) / (res["max_n_requirements"] / res["n_policy_types"])
                )

            to_plot["x"].append(n_req * int(res["n_policy_types"]))
            to_plot["y"].append(statistics.mean(it_acc) if len(it_acc) >= 1 else it_acc[0])
            to_plot["min_y"].append(min(it_acc))
            to_plot["max_y"].append(max(it_acc))

        if len(to_plot["x"]) > 0:
            plot_accuracy = True
            model_params = model2plot[model]
            plt.plot(
                to_plot["x"], to_plot["y"],
                marker=model_params["marker"],
                fillstyle='none',
                linestyle='--',
                color=model_params["color"],
                label=model_params["label"]
            )

            for idx, x in enumerate(to_plot['x']):
                plt.errorbar(
                    x,
                    to_plot['y'][idx],
                    yerr=[[to_plot['y'][idx] - to_plot['min_y'][idx]],
                          [to_plot['max_y'][idx] - to_plot['y'][idx]]],
                    color=model_params["color"],
                    elinewidth=1, capsize=1
                )

            if x_ticks is None:
                x_ticks = to_plot["x"]

    if plot_accuracy:
        plt.ylim([-0.1, 1.2])
        plt.yticks(np.arange(0, 1.2, 0.25))
        plt.xscale('log', base=10)
        plt.xticks(x_ticks)
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.legend(loc="lower left", labelspacing=0.1, ncol=1, prop={'size': 9})
        plt.xlabel('Batch Size')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.savefig(
            os.path.join(figures_path, f"step_1_fn_accuracy-function-call-{requirements_str}-adhoc.pdf"),
            format="pdf", bbox_inches='tight'
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, required=False, default="results_function_call")
    parser.add_argument('--figures_path', type=str, required=True)

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    plt.figure(figsize=(4, 2))
    matplotlib.rc('font', size=10)
    matplotlib.rcParams['hatch.linewidth'] = 0.3
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    os.makedirs(args.figures_path, exist_ok=True)

    _plot(args.results_path, args.figures_path, SortedSet({"reachability"}))
    _plot(args.results_path, args.figures_path, SortedSet({"reachability", "waypoint"}))
    _plot(args.results_path, args.figures_path, SortedSet({"loadbalancing", "reachability", "waypoint"}))


if __name__ == "__main__":
    main(parse_args())
