import argparse
import os
import statistics
import sys

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sortedcontainers import SortedSet

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from extractor.data_extractor import step_1_function_call_extract

requirements2plot = {
    "reachability": {
        "label": "1 Policy",
        "color": "#377eb8",
        "marker": "s"
    },
    "reachability_waypoint": {
        "label": "2 Policies",
        "color": "#984ea3",
        "marker": "^"
    },
    "loadbalancing_reachability_waypoint": {
        "label": "3 Policies",
        "color": "#a65628",
        "marker": "o"
    },
}


def _plot(results_path: str, figures_path: str, model_name: str, function_call_type: str) -> None:
    requirements2result = {}

    reqs = [{"reachability"}, {"reachability", "waypoint"}, {"reachability", "waypoint", "loadbalancing"}]

    # Accuracy
    for requirements in reqs:
        requirements_str = "_".join(SortedSet(requirements))

        requirements2result[requirements_str] = step_1_function_call_extract(
            results_path, requirements, function_call_type, model_name, "accuracy"
        )

    ax = plt.gca()

    for requirements_str, results in requirements2result.items():
        if not results:
            continue

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

        model_params = requirements2plot[requirements_str]

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

    plt.ylim([-0.1, 1.2])
    plt.yticks(np.arange(0, 1.2, 0.25))
    plt.xscale('log', base=10)
    plt.xticks([1, 2, 5, 10, 20, 50, 100])
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.legend(loc="lower left", labelspacing=0.1, ncol=1)
    plt.xlabel('Batch Size')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(
        os.path.join(figures_path, f"step_1_fn_accuracy-{model_name}-function-call-{function_call_type}.pdf"),
        format="pdf", bbox_inches='tight'
    )

    # Cost
    for requirements in reqs:
        requirements_str = "_".join(SortedSet(requirements))

        requirements2result[requirements_str] = step_1_function_call_extract(
            results_path, requirements, function_call_type, [model_name], "cost"
        )
    plot_costs = False

    plt.clf()
    ax = plt.gca()
    for requirements_str, results in requirements2result.items():
        if not results:
            continue

        to_plot = {"x": [], "y": [], "min_y": [], "max_y": []}
        for n_req, res in results.items():
            it_cost = []
            for cost in res["data"].values():
                it_cost.append(
                    statistics.mean([x for x in cost if x > 0]) if len(cost) > 0 else 0
                )
            it_cost = [x for x in it_cost if x > 0]

            to_plot["x"].append(n_req * int(res["n_policy_types"]))
            to_plot["y"].append(statistics.mean(it_cost) if len(it_cost) > 0 else 0)
            to_plot["min_y"].append(min(it_cost) if len(it_cost) > 0 else 0)
            to_plot["max_y"].append(max(it_cost) if len(it_cost) > 0 else 0)

        if any([x > 0 for x in to_plot["y"]]):
            plot_costs = True
            model_params = requirements2plot[requirements_str]

            to_plot["y"] = list(filter(lambda x: x > 0, to_plot["y"]))
            to_plot["x"] = to_plot["x"][0:len(to_plot["y"])]

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

    if plot_costs:
        plt.xscale('log', base=10)
        plt.xticks([1, 2, 5, 10, 20, 50, 100])
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.legend(loc="lower left", labelspacing=0.1, ncol=1)
        plt.xlabel('Batch Size')
        plt.ylabel('Cost [$]')
        plt.yscale('log', base=10)
        plt.ylim(0.0003, 0.005)
        plt.yticks(
            [0.0003, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005],
            [
                '$3\\times10^{-4}$', '$4\\times10^{-4}$', '$10^{-3}$', '$2\\times10^{-3}$', '$3\\times10^{-3}$',
                None, None
            ]
        )
        plt.grid(True)
        plt.savefig(
            os.path.join(figures_path, f"step_1_fn_cost-{model_name}-function-call-{function_call_type}.pdf"),
            format="pdf", bbox_inches='tight'
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, required=False, default="results_function_call")
    parser.add_argument('--figures_path', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--type', type=str, required=True, choices=['native', 'adhoc'])

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    plt.figure(figsize=(4, 2))
    mpl.rc('font', size=10)
    mpl.rcParams['hatch.linewidth'] = 0.3
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    os.makedirs(args.figures_path, exist_ok=True)

    _plot(args.results_path, args.figures_path, args.model, args.type)


if __name__ == "__main__":
    main(parse_args())
