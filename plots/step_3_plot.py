import argparse
import csv
import difflib
import glob
import json
import os
import re
import statistics

import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def extract_result(file_path: str) -> (dict, dict):
    average = {}
    mode = None

    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for res in reader:
            scenario_name = res["scenario_name"]
            if scenario_name not in average:
                average[scenario_name] = {}

            if not mode:
                mode = res["mode"]

            if mode == 'rag':
                mode += '_' + res["chunk_size"] if "fusion" not in res else '_fusion'

            it = int(res["iteration"])
            average[scenario_name][it] = json.loads(res["result"])

            for dev_name, outputs in average[scenario_name][it].items():
                if "diff_similarity" not in average[scenario_name][it][dev_name]:
                    expected = outputs["expected"].replace('!', '')
                    expected = [re.sub(r"^ +", "", x) for x in expected.split('\n')]
                    expected = [x for x in expected if x]
                    generated = outputs["generated"].replace('!', '')
                    generated = [re.sub(r"^ +", "", x) for x in generated.split('\n')]
                    generated = [x for x in generated if x]

                    seq_matcher = difflib.SequenceMatcher(lambda x: "[PLACEHOLDER]" in x, generated, expected)

                    average[scenario_name][it][dev_name]["diff_similarity"] = seq_matcher.ratio()

    return mode, average


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

scenario2params = {
    "ospf_simple": {
        "color": "#377eb8",
        "marker": "o"
    },
    "ospf_multiarea": {
        "color": "#4daf4a",
        "marker": "^"
    },
    "rip": {
        "color": "#e41a1c",
        "marker": "v"
    },
    "bgp_simple": {
        "color": "#984ea3",
        "marker": "s"
    },
    "rift_dc": {
        "color": "#a65628",
        "marker": "<"
    },
}


def plot(figures_path: str, results: dict) -> None:
    to_plot_bars = {}
    for mode in mode2params.keys():
        real_mode = 'rag_9000' if mode == 'rag' else mode

        for scenario, it2avg in results[real_mode].items():
            it_sim = []
            with_failures = set()
            total_runs = len(it2avg.keys())
            for it_num, it_res in it2avg.items():
                for name, res in it_res.items():
                    if res['n_daemon_errors'] > 0:
                        with_failures.add(it_num)
                        break

            for it_num, it_res in [x for x in it2avg.items() if x[0] not in with_failures]:
                values = [x["diff_similarity_daemon"] for x in it_res.values()]
                it_sim.append(
                    statistics.mean(values)
                )

            if mode not in to_plot_bars:
                to_plot_bars[mode] = {"x": [], "y": [], "min_y": [], "max_y": [], "success": []}
            to_plot_bars[mode]["x"].append(scenario)
            to_plot_bars[mode]["y"].append(statistics.mean(it_sim) if it_sim else 0)
            to_plot_bars[mode]["min_y"].append(min(it_sim) if it_sim else 0)
            to_plot_bars[mode]["max_y"].append(max(it_sim) if it_sim else 0)
            to_plot_bars[mode]["success"].append(str(total_runs - len(with_failures)))

    base_figures_path = os.path.join(".", figures_path)
    os.makedirs(base_figures_path, exist_ok=True)

    plt.clf()
    plt.figure(figsize=(4.5, 2))

    x = np.arange(5)
    width = 0.15
    multiplier = 0

    for mode, to_plot in to_plot_bars.items():
        params = mode2params[mode]

        real_lbl = mode2label[mode] if 'rag' not in mode else mode2label['rag'] + f" (Chunk Size 9000)"

        offset = width * multiplier
        vbar = plt.bar(x + offset, to_plot["y"], width=width, label=real_lbl,
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

    plt.savefig(os.path.join(figures_path, f"step_3.pdf"), format="pdf", bbox_inches='tight')


def plot_by_rag(figures_path: str, results: dict) -> None:
    raglbl2chunk = {}
    for k in results.keys():
        if 'rag' in k and 'fusion' not in k:
            (_, chunk) = k.split('_')
            raglbl2chunk[k] = int(chunk)

    raglbl2chunk = dict(sorted([(x, y) for x, y in raglbl2chunk.items()], key=lambda x: x[1]))
    to_plot_lines = {}

    for lbl, chunk in raglbl2chunk.items():
        for scenario, it2avg in results[lbl].items():
            it_sim = []
            with_failures = set()
            total_runs = len(it2avg.keys())
            for it_num, it_res in it2avg.items():
                for name, res in it_res.items():
                    if res['n_daemon_errors'] > 0:
                        with_failures.add(it_num)
                        break

            for it_num, it_res in [x for x in it2avg.items() if x[0] not in with_failures]:
                values = [x["diff_similarity_daemon"] for x in it_res.values()]
                it_sim.append(
                    statistics.mean(values)
                )

            if scenario not in to_plot_lines:
                to_plot_lines[scenario] = {"x": [], "y": [], "min_y": [], "max_y": []}
            to_plot_lines[scenario]["x"].append(chunk)
            to_plot_lines[scenario]["y"].append(statistics.mean(it_sim) if it_sim else 0)
            to_plot_lines[scenario]["min_y"].append(min(it_sim) if it_sim else 0)
            to_plot_lines[scenario]["max_y"].append(max(it_sim) if it_sim else 0)

    base_figures_path = os.path.join(".", figures_path)
    os.makedirs(base_figures_path, exist_ok=True)

    plt.clf()
    plt.figure(figsize=(4, 2))

    for scenario, to_plot in to_plot_lines.items():
        params = scenario2params[scenario]

        plt.plot(
            to_plot["x"], to_plot["y"], marker=params["marker"],
            fillstyle='none', color=params["color"], linestyle='--', label=scenario2label[scenario]
        )

        for idx, y in enumerate(to_plot["y"]):
            plt.errorbar(
                to_plot["x"][idx],
                y,
                yerr=[[y - to_plot['min_y'][idx]], [to_plot['max_y'][idx] - y]],
                color=params["color"],
                elinewidth=1, capsize=1
            )

    plt.xlabel('Chunk Size')
    plt.ylabel("Similarity Ratio")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), labelspacing=0.1, ncols=3)
    plt.xticks(range(1000, 11000, 1000), labels=list(map(lambda x: str(int(x / 1000)) + "k", range(1000, 11000, 1000))))
    plt.yticks(np.arange(0, 1.1, 0.1))

    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True, axis="y")

    plt.savefig(os.path.join(figures_path, f"step_3_rag.pdf"), format="pdf", bbox_inches='tight')


def plot_rag_fusion(figures_path: str, results: dict) -> None:
    to_plot_bars = {}
    for mode in mode2params.keys():
        real_mode = 'rag_fusion' if mode == 'rag' else mode
        for scenario, it2avg in results[real_mode].items():
            it_sim = []
            with_failures = set()
            total_runs = len(it2avg.keys())
            for it_num, it_res in it2avg.items():
                for name, res in it_res.items():
                    if res['n_daemon_errors'] > 0:
                        with_failures.add(it_num)
                        break

            for it_num, it_res in [x for x in it2avg.items() if x[0] not in with_failures]:
                values = [x["diff_similarity_daemon"] for x in it_res.values()]
                it_sim.append(
                    statistics.mean(values)
                )

            if mode not in to_plot_bars:
                to_plot_bars[mode] = {"x": [], "y": [], "min_y": [], "max_y": [], "success": []}
            to_plot_bars[mode]["x"].append(scenario)
            to_plot_bars[mode]["y"].append(statistics.mean(it_sim) if it_sim else 0)
            to_plot_bars[mode]["min_y"].append(min(it_sim) if it_sim else 0)
            to_plot_bars[mode]["max_y"].append(max(it_sim) if it_sim else 0)
            to_plot_bars[mode]["success"].append(str(total_runs - len(with_failures)) + "/" + str(total_runs))

    base_figures_path = os.path.join(".", figures_path)
    os.makedirs(base_figures_path, exist_ok=True)

    plt.clf()
    plt.figure(figsize=(4.5, 2))

    x = np.arange(5)
    width = 0.15
    multiplier = 0

    for mode, to_plot in to_plot_bars.items():
        params = mode2params[mode]

        real_lbl = mode2label[mode] if 'rag' not in mode else mode2label['rag'] + " (Fusion)"

        offset = width * multiplier
        vbar = plt.bar(x + offset, to_plot["y"], width=width, label=real_lbl,
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

        plt.bar_label(
            vbar,
            labels=[to_plot['success'][idx] for idx, _ in enumerate(to_plot["y"])],
            color=params["edgecolor"],
            fontsize=4,
            padding=1.5
        )

    plt.xlabel('Network Scenario')
    plt.ylabel("Similarity Ratio")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), labelspacing=0.1, ncols=2)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylim([0, 1.1])

    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True, axis="y")

    plt.savefig(os.path.join(figures_path, f"step_rag_fusion.pdf"), format="pdf", bbox_inches='tight')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, required=False, default="result")
    parser.add_argument('--figures_path', type=str, required=True)

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    mode2avg = {}
    results_files_list = glob.glob(os.path.join(".", args.results_path, "*.csv"))
    if results_files_list:
        for results_file in results_files_list:
            print(f"Working on {results_file}...")
            mode, avg = extract_result(results_file)
            mode2avg[mode] = avg

    matplotlib.rc('font', size=9)
    matplotlib.rcParams['hatch.linewidth'] = 0.3
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    plot(args.figures_path, mode2avg)
    plot_by_rag(args.figures_path, mode2avg)
    plot_rag_fusion(args.figures_path, mode2avg)


if __name__ == "__main__":
    main(parse_args())
