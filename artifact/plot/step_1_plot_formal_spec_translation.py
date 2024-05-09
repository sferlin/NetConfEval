import argparse
import os.path
import statistics
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sortedcontainers import SortedSet

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from extractor.data_extractor import step_1_translation_extract


def _plot(results_path: str, figures_path: str, requirements: SortedSet, models: str) -> None:
    if models == "gpt":
        model2plot = {
            "gpt-4-1106": {
                "label": "GPT-4-Turbo",
                "color": "#377eb8",
                "marker": "o"
            },
            "gpt-3.5-finetuned": {
                "label": "GPT-3.5-FT",
                "color": "#a65628",
                "marker": "s"
            },
            "gpt-3.5-0613": {
                "label": "GPT-3.5-Turbo",
                "color": "#984ea3",
                "marker": ">"
            },
        }
    elif models == "codellama":
        model2plot = {
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
    else:
        print(f"Models `{models}` not supported!")
        exit(1)

    requirements_str = "_".join(requirements)

    res_accuracy = {}
    # Accuracy
    for model_name in model2plot.keys():
        res_accuracy[model_name] = step_1_translation_extract(results_path, requirements, model_name, "accuracy")

    ax = plt.gca()
    x_ticks = None
    for model, results in res_accuracy.items():
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
        os.path.join(figures_path, f"step_1_formal_spec_accuracy-{models}-{requirements_str}.pdf"),
        format="pdf", bbox_inches='tight'
    )

    # Cost
    res_cost = {}
    for model_name in model2plot.keys():
        res_cost[model_name] = step_1_translation_extract(results_path, requirements, model_name, "cost")
    plot_costs = False

    plt.clf()
    ax = plt.gca()
    for model, results in res_cost.items():
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
            model_params = model2plot[model]

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
        plt.xticks(x_ticks)
        plt.ylim(0.00001, 0.02)
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.legend(loc='lower left', labelspacing=0.1, ncol=1, prop={'size': 9})
        plt.xlabel('Batch Size')
        plt.ylabel('Cost [$]')
        plt.yscale('log', base=10)
        plt.grid(True)
        plt.savefig(
            os.path.join(figures_path, f"step_1_formal_spec_cost-{models}-{requirements_str}.pdf"),
            format="pdf", bbox_inches='tight'
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, required=False, default="results_spec_translation")
    parser.add_argument('--figures_path', type=str, required=True)
    parser.add_argument('--policy_types', choices=["reachability", "waypoint", "loadbalancing"],
                        required=True, nargs='+')
    parser.add_argument("--models", type=str, choices=["gpt", "codellama"])

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    plt.figure(figsize=(4, 2))
    matplotlib.rc('font', size=10)
    matplotlib.rcParams['hatch.linewidth'] = 0.3
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    os.makedirs(args.figures_path, exist_ok=True)

    _plot(args.results_path, args.figures_path, SortedSet(args.policy_types), args.models)


if __name__ == "__main__":
    main(parse_args())
