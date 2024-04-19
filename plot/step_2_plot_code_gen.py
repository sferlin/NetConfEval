import argparse
import csv
import glob
import os
import re
import statistics
import sys

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

file2plot = [
    {
        "edgecolor": "#e41a1c",
        "hatch": "/////"
    },
    {
        "edgecolor": "#377eb8",
        "hatch": "\\\\\\\\\\"
    },
    {
        "edgecolor": "#4daf4a",
        "hatch": "----"
    },
    {
        "edgecolor": "#984ea3",
        "hatch": "xxxx"
    },
    {
        "edgecolor": "magenta",
        "hatch": "."
    },
    {
        "edgecolor": "cyan",
        "hatch": "o"
    }
]

key2xlabel = {
    "shortest_path": "Shortest Path",
    "reachability": "Reachability",
    "waypoint": "Waypoint",
    "loadbalancing": "Load-Balancing"
}


def extract_result(file):
    print(file)
    average = {}

    with open(os.path.join(file), 'r') as f:
        reader = csv.DictReader(f)
        for res in reader:
            if res["policy"]:
                policy = res["policy"]
                if policy not in average:
                    average[policy] = {}
                    average[policy]["time"] = []
                    average[policy]["feedback_num"] = []
                    average[policy]["format_error_num"] = []
                    average[policy]["syntax_error_num"] = []
                    average[policy]["test_error_num"] = []
                    average[policy]["total_cost"] = []

                average[policy]["time"].append(float(res["time"]))
                average[policy]["feedback_num"].append(float(res["feedback_num"]))
                average[policy]["format_error_num"].append(float(res["format_error_num"]))
                average[policy]["syntax_error_num"].append(float(res["syntax_error_num"]))
                average[policy]["test_error_num"].append(float(res["test_error_num"]))
                average[policy]["total_cost"].append(float(res["total_cost"]) if "total_cost" in res else 0)

    result = {}

    # print("policy\t\t\t", "time\t\t", "feedback\t", "failure\t", "cost")
    for policy, avg in average.items():
        result[policy] = {}
        result[policy]["average_time"] = statistics.mean(avg["time"])
        result[policy]["max_time"] = max(avg["time"])
        result[policy]["min_time"] = min(avg["time"])
        result[policy]["average_feedback"] = statistics.mean(avg["feedback_num"])
        result[policy]["max_feedback"] = max(avg["feedback_num"])
        result[policy]["min_feedback"] = min(avg["feedback_num"])
        result[policy]["average_success"] = statistics.mean([i < 10 for i in avg["feedback_num"]])
        result[policy]["average_cost"] = statistics.mean(avg["total_cost"])
        result[policy]["max_cost"] = max(avg["total_cost"])
        result[policy]["min_cost"] = min(avg["total_cost"])

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, required=False, default="result")
    parser.add_argument('--figures_path', type=str, required=True)

    return parser.parse_args()


def main(args: argparse.Namespace):
    files = []

    print(os.path.join(args.results_path, f"code-gpt-4-*.csv"))

    results_files_list = glob.glob(os.path.join('../', args.results_path, f"code-gpt-4-*.csv"))

    # print(results_files_list)
    # exit()

    for file in results_files_list:

        res = extract_result(file)
        if "basic-with_feedback" in file:
            label = "w/ Instruction w/ Feedback"
        elif "basic-without_feedback" in file:
            label = "w/ Instruction w/o Feedback"
        elif "no_detail-with_feedback" in file:
            label = "w/o Instruction w/ Feedback"
        elif "no_detail-without_feedback" in file:
            label = "w/o Instruction w/o Feedback"

        # match = re.match(".*-(gpt-4-.*)-1.*\.csv", file)
        # name = match.group(1)
        # label = ""
        # if name == "gpt-4-turbo-basic" or name == "gpt-4-basic":
        #     label = "w/ Instruction w/ Feedback"
        # elif name == "gpt-4-turbo-basic-without_feedback" or name == "gpt-4-without_feedback":
        #     label = "w/ Instruction w/o Feedback"
        # elif name == "gpt-4-turbo-without_detail" or name == "gpt-4-without_detail":
        #     label = "w/o Instruction w/ Feedback"
        # elif name == "gpt-4-turbo-without_detail-without_feedback" or name == "gpt-4-without_detail_without_feedback":
        #     label = "w/o Instruction w/o Feedback"
        files.append({"legend": label, "res": res})

    files.sort(key=lambda x: len(x["legend"]), reverse=True)

    bar_nums = len(files)
    labels_basic = list(files[0]["res"].keys())

    for i in range(1, bar_nums):
        labels_basic = [val for val in list(files[i]["res"].keys()) if val in labels_basic]

    x = np.arange(len(labels_basic))
    plt.clf()
    figure, axis = plt.subplots(1, 3, sharex=True, figsize=(12, 3))
    figure.text(0.5, -0.03, 'Policy', ha='center')


    plt.sca(axis[0])

    for i in range(0, bar_nums):
        file = files[i]
        success_rates = [file["res"][label]["average_success"] for label in labels_basic]

        plt.bar(x + 0.1 * (i - bar_nums / 2), success_rates, label=file["legend"], color="white",
                edgecolor=file2plot[i]["edgecolor"],
                hatch=file2plot[i]["hatch"], width=0.1)

    plt.xticks(range(0, len(labels_basic)), labels=[key2xlabel[val] for val in labels_basic], fontsize=8)
    plt.ylabel("Success Rate [%]")
    figure.legend(loc="upper center", labelspacing=0.1, ncol=2, bbox_to_anchor=(0.5, 1.13))
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True, axis="y")

    os.makedirs(args.figures_path, exist_ok=True)

    plt.sca(axis[1])
    for i in range(0, bar_nums):
        file = files[i]
        avg_feedbacks = [file["res"][label]["average_feedback"] for label in labels_basic]
        max_feedbacks = [file["res"][label]["max_feedback"] for label in labels_basic]
        min_feedbacks = [file["res"][label]["min_feedback"] for label in labels_basic]

        plt.bar(x + 0.1 * (i - bar_nums / 2), avg_feedbacks, label=file["legend"], color="white",
                edgecolor=file2plot[i]["edgecolor"],
                hatch=file2plot[i]["hatch"], width=0.1)

    plt.xticks(range(0, len(labels_basic)), labels=[key2xlabel[val] for val in labels_basic], fontsize=8)
    plt.ylabel("N. Attempts")
    plt.ylim(0, 11)
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True, axis="y")

    plt.sca(axis[2])
    for i in range(0, bar_nums):
        file = files[i]
        avg_costs = [file["res"][label]["average_cost"] for label in labels_basic]
        max_costs = [file["res"][label]["max_cost"] for label in labels_basic]
        min_costs = [file["res"][label]["min_cost"] for label in labels_basic]

        plt.bar(x + 0.1 * (i - bar_nums / 2), avg_costs, label=file["legend"], color="white",
                edgecolor=file2plot[i]["edgecolor"],
                hatch=file2plot[i]["hatch"], width=0.1)
    plt.xticks(range(0, len(labels_basic)), labels=[key2xlabel[val] for val in labels_basic], fontsize=8)
    plt.ylabel("Cost [$]")
    plt.ylim(0, 0.70)
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True, axis="y")

    figure.tight_layout()

    figure.savefig(os.path.join(args.figures_path, f"figure-step2-gpt4.pdf"),
                format="pdf",
                bbox_inches='tight')


if __name__ == "__main__":
    matplotlib.rc('font', size=10)
    matplotlib.rcParams['hatch.linewidth'] = 0.3
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    main(parse_args())