import argparse
import json
from pathlib import Path

import numpy as np

from utils.visualization_utils import plot_results

BASE_DIR = Path(__file__).parent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", help="Run ID to plot results for (e.g. 0505_A4)")
    args = parser.parse_args()
    plot_results(args.run_id)
