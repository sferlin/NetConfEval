import argparse
import csv
import os.path
import statistics
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['gpt-4', 'gpt-3.5', 'fine-tuned', 'bard', 'llama2-7b'], required=True)
    parser.add_argument('--file', type=str, required=True)
    return parser.parse_args()


def main(args):
    average = {
    }

    with open(os.path.join("evaluation", args.file), 'r') as file:
        reader = csv.DictReader(file)
        for res in reader:
            if res["n_requirements"] and res["n_requirements"] != '0':
                n_req = res["n_requirements"]
                if n_req not in average:
                    average[n_req] = {}
                    average[n_req]["time"] = []
                    average[n_req]["total"] = []
                    average[n_req]["success"] = []
                    average[n_req]["fail"] = []
                    average[n_req]["wrong"] = []
                    average[n_req]["accuracy"] = []
                    average[n_req]["total_cost"] = []
                    average[n_req]["count"] = 0

                # to filter some strange value
                if float(res["time"]) < 400:
                    average[n_req]["time"].append(float(res["time"]) / int(n_req))
                average[n_req]["total"].append(float(res["total"]))
                average[n_req]["success"].append(float(res["success"]))
                average[n_req]["fail"].append(float(res["fail"]))
                average[n_req]["wrong"].append(float(res["wrong"]))
                average[n_req]["accuracy"].append(float(res["accuracy"]))
                average[n_req]["total_cost"].append(float(res["total_cost"]) / int(n_req))
                average[n_req]["count"] += 1

    req = []
    time = []
    stdev_times = []
    accuracy = []
    stdev_accuracies = []
    total_cost = []
    stdev_total_cost = []

    for n_req, avg in average.items():
        average_time = statistics.mean(avg["time"])
        min_time = np.percentile(avg["time"], 0)
        max_time = np.percentile(avg["time"], 100)

        average_accuracy = statistics.mean(avg["time"])
        min_accuracy = np.percentile(avg["time"], 0)
        max_accuracy = np.percentile(avg["time"], 100)

        average_cost = statistics.mean(avg["total_cost"])
        min_cost = np.percentile(avg["total_cost"], 0)
        max_cost = np.percentile(avg["total_cost"], 100)

        average_accuracy = statistics.mean(avg["accuracy"])
        min_cost = np.percentile(avg["accuracy"], 0)
        max_cost = np.percentile(avg["accuracy"], 100)

        req.append(n_req)
        time.append(average_time)
        stdev_times.append([[average_time - min_time], [max_time - average_time]])
        total_cost.append(average_cost)
        accuracy.append(average_accuracy)
        stdev_accuracies.append([[average_accuracy - min_accuracy], [max_accuracy - average_accuracy]])
        stdev_total_cost.append([[average_cost - min_cost], [max_cost - average_cost]])

    # sub_axix = filter(lambda x:x%200 == 0, x_axix)
    print(req)
    # print(time, stdev_times)
    print(total_cost, stdev_total_cost)
    # print(accuracy, stdev_accuracies)

    plt.figure(figsize=(4, 2))
    mpl.rc('font', size=8)
    mpl.rcParams['hatch.linewidth'] = 0.3
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    ax = plt.plot()

    plt.plot(req, accuracy, marker='o', fillstyle='none', linestyle='--', color='red', label='Accuracy')
    plt.ylim([-0.1, 1.4])
    plt.yticks(np.arange(0, 1.2, 0.25))
    # for a, b in zip(req, accuracy):
    # plt.text(a, b, round(b, 3), ha='center', va='bottom', fontsize=8)
    plt.legend(loc='upper left', labelspacing=0.2, prop={'size': 8})
    plt.xlabel('Number of Requirements')
    plt.ylabel('Accuracy', fontdict={'color': 'red', 'weight': 'bold'})
    plt.grid(True)

    # for idx, x in enumerate(req):
    #     plt.errorbar(x, accuracy[idx], yerr=stdev_accuracies[idx], color='darkred', elinewidth=1, capsize=1)

    ax2 = plt.twinx()

    plt.plot(req, [1000 * i for i in total_cost], marker='^', fillstyle='none', color='#3182bd', linestyle='--',
             label='Price/Requirement ($/1000)')
    plt.plot(req, time, marker='^', fillstyle='none', color='#9ecae1', linestyle='--', label='Time/Requirement (s)')
    plt.yticks(np.arange(0, 12.5, 2.5))
    plt.ylim([-1, 14])

    # for idx, x in enumerate(req):
    #     plt.errorbar(x, total_cost[idx], yerr=stdev_total_cost[idx], color='#08306b', elinewidth=1, capsize=1)
    #     plt.errorbar(x, time[idx], yerr=stdev_times[idx], color='#08519c', elinewidth=1, capsize=1)

    # for a, b in zip(req, money_cost):
    #     plt.text(a, b, round(b, 3), ha='center', va='bottom', fontsize=8)
    # for a, b in zip(req, time):
    #     plt.text(a, b, round(b, 3), ha='center', va='bottom', fontsize=8)

    plt.legend(loc='upper right', labelspacing=0.2, prop={'size': 8})
    plt.ylabel('Cost', fontdict={'color': '#3182bd', 'weight': 'bold'})

    plt.savefig(os.path.join("evaluation", f"accuracy_cost_gpt_{args.model}.pdf"), format="pdf", bbox_inches='tight')
    # plt.show()

    # plt.savefig('figure_complexity.pdf')


if __name__ == "__main__":
    main(parse_args())
