import os
import sys
import matplotlib.pyplot as plt
import argparse
env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from pytracking.evaluation import Tracker, get_dataset, trackerlist


def main():
    parser = argparse.ArgumentParser(description='Generate success and precision plots')
    parser.add_argument('tracker_name', type=str)
    parser.add_argument('tracker_param', type=str)
    parser.add_argument('--run_id', type=int, default=None)
    parser.add_argument('--dataset', type=str)

    args = parser.parse_args()

    trackers = []
    display_name = args.tracker_param + str(args.run_id)
    trackers.extend(trackerlist(args.tracker_name, args.tracker_param, args.run_id, display_name))

    dataset = get_dataset(args.dataset)

    report_name = args.tracker_param + '_' + args.dataset
    plot_results(trackers, dataset, report_name, merge_results=True, plot_types=('success', 'prec'), 
             skip_missing_seq=True, force_evaluation=True, plot_bin_gap=0.05, 
             exclude_invalid_frames=False)

if __name__ == '__main__':
    main()

