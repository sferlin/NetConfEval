import argparse
import os
import statistics
import sys

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from extractor.data_extractor import step_2_code_gen_extract
from netconfeval.common.model_configs import model_configurations

types2plot = {
    "basic-with_feedback": {
        "label": "w/ Instruction w/ Feedback",
        "edgecolor": "#e41a1c",
        "hatch": "/////"
    },
    "basic-without_feedback": {
        "label": "w/ Instruction w/o Feedback",
        "edgecolor": "#377eb8",
        "hatch": "\\\\\\\\\\"
    },
    "no_detail-with_feedback": {
        "label": "w/o Instruction w/ Feedback",
        "edgecolor": "#4daf4a",
        "hatch": "----"
    },
    "no_detail-without_feedback": {
        "label": "w/o Instruction w/o Feedback",
        "edgecolor": "#984ea3",
        "hatch": "xxxx"
    },
}

x_labels = {
    "shortest_path": "Shortest Path",
    "reachability": "Reachability",
    "waypoint": "Waypoint",
    "loadbalancing": "Load-Balancing"
}


def _plot(results_path: str, figures_path: str, model: str) -> None:
    results = {}
    for prompt in ["basic", "no_detail"]:
        for feedback in ["with_feedback", "without_feedback"]:
            results[f"{prompt}-{feedback}"] = step_2_code_gen_extract(results_path, prompt, feedback, model)

    plt.clf()
    figure, axis = plt.subplots(1, 3, sharex=True, figsize=(12, 3))
    figure.text(0.5, -0.03, 'Policy', ha='center')

    plt.sca(axis[0])
    x_ticks = None
    for x, (exp_type, res) in enumerate(results.items()):
        type_config = types2plot[exp_type]
        bar_nums = len(res)

        if x_ticks is None:
            x_ticks = res.keys()

        for idx, data in enumerate(res.values()):
            avg_succ = statistics.mean([i < 10 for i in data["feedback_num"]])
            plt.bar(
                idx + 0.1 * (x - bar_nums / 2), avg_succ, label=type_config["label"], color="white",
                edgecolor=type_config["edgecolor"], hatch=type_config["hatch"], width=0.1
            )

    plt.xticks(range(0, len(x_ticks)), labels=[x_labels[val] for val in x_ticks], fontsize=8)
    plt.ylabel("Success Rate [%]")

    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True, axis="y")

    plt.sca(axis[1])
    x_ticks = None
    for x, (exp_type, res) in enumerate(results.items()):
        type_config = types2plot[exp_type]
        bar_nums = len(res)

        if x_ticks is None:
            x_ticks = res.keys()

        for idx, data in enumerate(res.values()):
            avg_feedback = statistics.mean(data["feedback_num"])
            plt.bar(
                idx + 0.1 * (x - bar_nums / 2), avg_feedback, label=type_config["label"], color="white",
                edgecolor=type_config["edgecolor"], hatch=type_config["hatch"], width=0.1
            )

    plt.xticks(range(0, len(x_ticks)), labels=[x_labels[val] for val in x_ticks], fontsize=8)
    plt.ylabel("N. Attempts")
    plt.ylim(0, 11)

    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True, axis="y")

    plt.sca(axis[2])
    x_ticks = None
    for x, (exp_type, res) in enumerate(results.items()):
        type_config = types2plot[exp_type]
        bar_nums = len(res)

        if x_ticks is None:
            x_ticks = res.keys()

        for idx, data in enumerate(res.values()):
            avg_cost = statistics.mean(data["total_cost"])
            plt.bar(
                idx + 0.1 * (x - bar_nums / 2), avg_cost, label=type_config["label"], color="white",
                edgecolor=type_config["edgecolor"], hatch=type_config["hatch"], width=0.1
            )

    plt.xticks(range(0, len(x_ticks)), labels=[x_labels[val] for val in x_ticks], fontsize=8)
    plt.ylabel("Cost [$]")
    plt.ylim(0, 0.70)

    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True, axis="y")
    figure.tight_layout()

    legend_patches = []
    for type_config in types2plot.values():
        legend_patches.append(
            mpatches.Patch(
                facecolor="white", edgecolor=type_config["edgecolor"], hatch=type_config["hatch"],
                label=type_config["label"]
            )
        )

    figure.legend(
        loc="upper center", handles=legend_patches,
        labelspacing=0.1, ncol=2, bbox_to_anchor=(0.5, 1.13)
    )
    figure.savefig(
        os.path.join(figures_path, f"step_2_code_gen-{model}.pdf"),
        format="pdf", bbox_inches='tight'
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, required=False, default="results_code_gen")
    parser.add_argument('--figures_path', type=str, required=True)
    parser.add_argument('--model', choices=list(model_configurations.keys()), required=True)

    return parser.parse_args()


def main(args: argparse.Namespace):
    matplotlib.rc('font', size=10)
    matplotlib.rcParams['hatch.linewidth'] = 0.3
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    os.makedirs(args.figures_path, exist_ok=True)

    _plot(args.results_path, args.figures_path, args.model)


if __name__ == "__main__":
    main(parse_args())
