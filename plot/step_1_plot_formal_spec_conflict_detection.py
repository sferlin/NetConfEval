import argparse
import os
import statistics
import sys

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sortedcontainers import SortedSet

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from extractor.data_extractor import step_1_conflict_detection_extract


def _plot(results_path: str, figures_path: str, is_combined: bool, requirements: SortedSet, metric: str) -> None:
    if not is_combined:
        model2plot = {
            "gpt-4-1106": {
                "label": "GPT-4-Turbo",
                "color": "#377eb8",
                "marker": "o"
            },
            "gpt-4": {
                "label": "GPT-4",
                "color": "#a65628",
                "marker": "s"
            },
            "gpt-3.5-0613": {
                "label": "GPT-3.5-Turbo",
                "color": "#984ea3",
                "marker": ">"
            }
        }
    else:
        model2plot = {
            "gpt-4": {
                "label": "GPT-4",
                "color": "#a65628",
                "marker": "s"
            },
            "gpt-4-1106-combined": {
                "label": "GPT-4 (Combined)",
                "color": "#984ea3",
                "marker": "<"
            },
        }

    requirements_str = "_".join(requirements)

    results = {}
    for model in model2plot.keys():
        results[model] = step_1_conflict_detection_extract(results_path, requirements, model, metric)

    ax = plt.gca()
    x_ticks = None
    for model, results in results.items():
        to_plot = {"x": [], "y": [], "min_y": [], "max_y": []}
        for n_req, res in results.items():
            to_plot["x"].append(n_req * int(res["n_policy_types"]))
            to_plot["y"].append(statistics.mean(res["data"]) if len(res["data"]) >= 1 else res["data"][0])
            to_plot["min_y"].append(min(res["data"]))
            to_plot["max_y"].append(max(res["data"]))

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

    plt.ylim([0, 1.2])
    plt.yticks(np.arange(0, 1.2, 0.25))
    plt.xscale('log', base=10)
    plt.xticks(x_ticks)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.legend(
        loc='lower left' if metric == "f1_score" else "lower right", labelspacing=0.2, ncol=1,
        prop={'size': 8}
    )
    plt.xlabel('Batch Size')
    label = "Accuracy" if metric == "accuracy" else "Recall" if metric == "recall" else "F1-Score" if metric == "f1_score" else "None"
    plt.ylabel(label)
    plt.grid(True)
    plt.savefig(
        os.path.join(figures_path,
                     f"step_1_detection-{requirements_str}-{metric}-{'combined' if is_combined else ''}.pdf"),
        format="pdf", bbox_inches='tight'
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, required=False, default="results_conflict_detection")
    parser.add_argument('--figures_path', type=str, required=True)
    parser.add_argument('--combined', action='store_true', required=False)
    parser.add_argument('--policy_types', choices=["reachability", "waypoint", "loadbalancing"],
                        required=True, nargs='+')
    parser.add_argument('--metric', type=str, required=True, choices=["accuracy", "recall", "f1_score"])

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    plt.figure(figsize=(4, 2))
    mpl.rc('font', size=10)
    mpl.rcParams['hatch.linewidth'] = 0.3
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    os.makedirs(args.figures_path, exist_ok=True)

    _plot(args.results_path, args.figures_path, args.combined, SortedSet(args.policy_types), args.metric)


if __name__ == "__main__":
    main(parse_args())
