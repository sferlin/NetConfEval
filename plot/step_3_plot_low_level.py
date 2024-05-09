import argparse
import os
import statistics
import sys

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from extractor.data_extractor import step_3_low_level_extract
from netconfeval.common.model_configs import model_configurations

mode2label = {
    "idx": "Index + Section",
    "none": "No Documentation",
    "rag": "RAG",
    "full": "Full Documentation"
}

scenario2label = {
    "ospf_simple": "OSPF",
    "ospf_multiarea": "OSPF (Multi)",
    "rip": "RIP",
    "bgp_simple": "BGP",
    "rift_dc": "RIFT"
}

mode2params = {
    "none": {
        "edgecolor": "#377eb8",
        "hatch": "/////"
    },
    "full": {
        "edgecolor": "#4daf4a",
        "hatch": "\\\\\\\\\\"
    },
    "idx": {
        "edgecolor": "#e41a1c",
        "hatch": "++++"
    },
    "rag": {
        "edgecolor": "#984ea3",
        "hatch": "----"
    }
}


def _plot(results_path: str, figures_path: str, model: str) -> None:
    results = {}
    for mode in mode2params.keys():
        rag_size = 9000 if mode == "rag" else None
        results[mode] = step_3_low_level_extract(results_path, model, mode, rag_size)

    to_plot_by_mode = {}
    for mode, res in results.items():
        for scenario, data in res.items():
            it_similarity = []
            with_failures = set()
            total_runs = len(data.keys())
            for it_num, it_res in data.items():
                for device_res in it_res.values():
                    if device_res['n_daemon_errors'] > 0:
                        with_failures.add(it_num)
                        break

            for it_num, it_res in [x for x in data.items() if x[0] not in with_failures]:
                values = [x["diff_similarity_daemon"] for x in it_res.values()]
                it_similarity.append(
                    statistics.mean(values)
                )

            if mode not in to_plot_by_mode:
                to_plot_by_mode[mode] = {"x": [], "y": [], "min_y": [], "max_y": [], "success": []}
            to_plot_by_mode[mode]["x"].append(scenario)
            to_plot_by_mode[mode]["y"].append(statistics.mean(it_similarity) if it_similarity else 0)
            to_plot_by_mode[mode]["min_y"].append(min(it_similarity) if it_similarity else 0)
            to_plot_by_mode[mode]["max_y"].append(max(it_similarity) if it_similarity else 0)
            to_plot_by_mode[mode]["success"].append(str(total_runs - len(with_failures)))

    plt.clf()

    x = np.arange(5)
    width = 0.15
    multiplier = 0

    for mode, to_plot in to_plot_by_mode.items():
        params = mode2params[mode]

        real_lbl = mode2label[mode] if 'rag' not in mode else mode2label['rag'] + f" (Chunk Size 9000)"

        offset = width * multiplier
        plt.bar(x + offset, to_plot["y"], width=width, label=real_lbl,
                edgecolor=params["edgecolor"], hatch=params["hatch"], color="white")
        multiplier += 1
        plt.xticks(x + width + width / 2, labels=list(map(lambda x: scenario2label[x], to_plot["x"])))

        for idx, y in enumerate(to_plot["y"]):
            plt.errorbar(
                idx + offset,
                y,
                yerr=[[y - to_plot['min_y'][idx]], [to_plot['max_y'][idx] - y]],
                color="black",
                elinewidth=1, capsize=1
            )

            plt.text(
                idx + offset,
                to_plot['max_y'][idx],
                to_plot['success'][idx],
                color=params["edgecolor"],
                fontsize=8,
                ha='center', va='bottom',
            )

    plt.xlabel('Network Scenario')
    plt.ylabel("Similarity Ratio")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), labelspacing=0.1, ncols=2)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylim([0, 1.1])

    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True, axis="y")

    plt.savefig(os.path.join(figures_path, f"step_3_low_level-{model}.pdf"), format="pdf", bbox_inches='tight')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, required=False, default="results_low_level")
    parser.add_argument('--figures_path', type=str, required=True)
    parser.add_argument('--model', choices=list(model_configurations.keys()), required=True)

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    plt.figure(figsize=(4.5, 2))
    matplotlib.rc('font', size=9)
    matplotlib.rcParams['hatch.linewidth'] = 0.3
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    os.makedirs(args.figures_path, exist_ok=True)

    _plot(args.results_path, args.figures_path, args.model)


if __name__ == "__main__":
    main(parse_args())
