import argparse
import os
import sys

import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sortedcontainers import SortedSet

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from extractor.data_extractor import step_1_conflict_distance_extract
from netconfeval.common.model_configs import model_configurations


def _plot(results_path: str, figures_path: str, requirements: SortedSet, model: str) -> None:
    requirements_str = "_".join(requirements)

    results = step_1_conflict_distance_extract(results_path, requirements, model)
    corr = np.corrcoef(results)
    mask = np.tril(np.full_like(corr, 0))

    ax = sns.heatmap(
        data=results, cmap="Blues", mask=mask,
        cbar_kws=dict(use_gridspec=False, label='N. of Detected Conflicts\n(Out of 10)')
    )
    ax.figure.axes[-1].yaxis.label.set_size(10)
    plt.xlabel('Distance')
    plt.ylabel('Index of \nSelected Requirement')

    plt.xlim(1, 34)
    plt.ylim(0, 33)

    fig = ax.get_figure()
    fig.savefig(
        os.path.join(figures_path, f"step_1_distance-{requirements_str}-{model}.pdf"),
        format="pdf", bbox_inches='tight'
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, required=False, default="results_conflict_distance")
    parser.add_argument('--figures_path', type=str, required=True)
    parser.add_argument('--policy_types', choices=["reachability", "waypoint", "loadbalancing"],
                        required=True, nargs='+')
    parser.add_argument('--model', choices=list(model_configurations.keys()), required=True)

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    plt.figure(figsize=(4, 3))
    matplotlib.rc('font', size=10)
    matplotlib.rcParams['hatch.linewidth'] = 0.3
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    os.makedirs(args.figures_path, exist_ok=True)

    _plot(args.results_path, args.figures_path, args.policy_types, args.model)


if __name__ == "__main__":
    main(parse_args())
