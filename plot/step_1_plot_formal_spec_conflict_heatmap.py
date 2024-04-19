import argparse
import csv
import glob

import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

def heatmap(data_loc: str, fig_path: str) -> None:
    results = [[0] * 34 for i in range(34)]

    corr = np.corrcoef(results)
    with open(data_loc, "r") as file:
        reader = csv.DictReader(file)
        for res in reader:
            # if int(res["iteration"]) == 0:
            if res["conflict_detect"] == "True":
                print(res["index_start"], res["index_end"])
                # print(res["conflict_detect"])
                print(res["index_start"])
                print(res["index_end"])
                results[int(res["index_start"])][int(res["index_end"]) - int(res["index_start"])] += 1

    for i in results:
        print(i)

    # pearson coefficients
    # lower triangle
    mask = np.tril(np.full_like(corr, 0))
    # mask = np.tril(np.zeros_like(corr))
    # plt.get_cmap("")

    # sns.set(font_scale = 1)
    plt.figure(figsize=(4, 3))
    ax = sns.heatmap(data=results, cmap="Blues", mask=mask,
                     cbar_kws=dict(use_gridspec=False, label='N. of Detected Conflicts\n(Out of 10)'))
    ax.figure.axes[-1].yaxis.label.set_size(10)
    plt.xlabel('Distance')
    plt.ylabel('Index of \nSelected Requirement')
    # ax.xaxis.tick_top()
    # ax.xaxis.set_label_position('top')
    # ax.yaxis.tick_right()
    # ax.yaxis.set_label_position('right')
    # plt.ylim(34, 0)

    plt.xlim(1, 34)
    plt.ylim(0, 33)

    fig = ax.get_figure()
    fig.savefig(fig_path, format="pdf", bbox_inches='tight')


def main(args: argparse.Namespace) -> None:
    matplotlib.rc('font', size=10)
    matplotlib.rcParams['hatch.linewidth'] = 0.3
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    results_files_list = glob.glob(os.path.join("../", args.results_path, f"*.csv"))
    base_figures_path = os.path.join("../plot", args.figures_path)
    os.makedirs(base_figures_path, exist_ok=True)

    for file in results_files_list:
        heatmap(
            file,
            os.path.join(base_figures_path, "heatmap-gpt-3.5.pdf")
        )
    # heatmap(
    #     "evaluation/result_conflict/result_distance/result-gpt-4-turbo-loadbalancing_reachability_waypoint-conflict-1700747740.0543778.csv",
    #     "evaluation/result_conflict/result_distance/heatmap-gpt-4.pdf"
    # )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, required=False, default="result")
    parser.add_argument('--figures_path', type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
