# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Module containing diverse utility functions."""
import logging
import pathlib
import subprocess
import time
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

logger = logging.getLogger(__name__)


class LRTensorBoard(tf.keras.callbacks.TensorBoard):
    """Custom TensorBoard callback that tracks the learning rate."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Keras 3: optimizer attribute is typically "learning_rate", older: "lr"
        lr_obj = getattr(self.model.optimizer, "learning_rate", None)
        if lr_obj is None:
            lr_obj = getattr(self.model.optimizer, "lr", None)

        try:
            if callable(lr_obj):
                lr_val = lr_obj(self.model.optimizer.iterations)
            else:
                lr_val = lr_obj

            if lr_val is None:
                raise ValueError("No learning rate available on optimizer.")

            # Convert to python float robustly (eager-safe)
            if hasattr(lr_val, "numpy"):
                lr_val = lr_val.numpy()
            lr_val = float(lr_val)

            logs.update({"lr": lr_val})
        except Exception as e:
            # Don't break training because of logging
            logger.debug(f"Could not log learning rate: {e}")

        super().on_epoch_end(epoch, logs)


def model_needs_retraining(
    seeds_path: pathlib.Path,
    timestamp_last_training: int,
    n_seeds_last_training: int,
    retraining_interval_s: int = 3600,
    n_new_seeds_for_retraining: int = 10,
) -> bool:
    """
    Determine whether the model needs retraining based on:
      * time since last training
      * number of new seeds since last training
    """
    n_current_seeds = len(list(seeds_path.glob("id*")))
    time_since_retrain = int(time.time()) - int(timestamp_last_training)
    return (
        time_since_retrain >= retraining_interval_s
        and n_current_seeds >= n_seeds_last_training + n_new_seeds_for_retraining
    )


def get_max_file_size(path: Union[pathlib.Path, str]) -> int:
    """
    Return the maximum file size in the given path.

    The folder is *not* scanned recursively.
    """
    p = pathlib.Path(path)
    files = [f for f in p.glob("*") if f.is_file()]
    if not files:
        raise ValueError(f"No files found in {p} to compute max file size.")
    return max(f.stat().st_size for f in files)


def create_work_folders(path: Union[str, pathlib.Path] = ".") -> None:
    """
    Create folder(s) for machine learning artifacts.
    """
    folders_to_create = ["models"]
    parent = pathlib.Path(path) if isinstance(path, str) else path
    for folder in folders_to_create:
        (parent / folder).mkdir(parents=True, exist_ok=True)


def get_timestamp_millis_from_filename(filename: str) -> int:
    """
    Extract AFL++ timestamp from queue filename (token "time:<ms>").
    """
    for token in filename.split(","):
        key_val = token.split(":")
        if key_val[0] == "time" and len(key_val) > 1:
            try:
                return int(key_val[1])
            except ValueError:
                return 0
    return 0


def _add_to_dict(data_dict, key, value):
    """Append value to its corresponding key in the dict without erasing existing values."""
    if key in data_dict:
        data_dict[key].append(value)
    else:
        data_dict[key] = [value]


def _search_afl_plot_data(
    folder: pathlib.Path, data_columns: List[str], plot_file: str
) -> Dict[str, pd.DataFrame]:
    """
    Search input folder for AFL++ plotting data. Each last folder in the path containing the
    plot data file is considered a trial; trials are grouped into experiments.
    """
    all_plot_data: Dict[str, pd.DataFrame] = {}
    all_plot_data_files = list(folder.glob(f"**/{plot_file}"))

    for plot_data in all_plot_data_files:
        cov_data = pd.read_csv(plot_data, sep=", ", usecols=data_columns, engine="python")
        if "default" in str(plot_data):
            experiment_key = str(plot_data.relative_to(folder).parent.parent.parent)
        else:
            experiment_key = str(plot_data.relative_to(folder).parent.parent)
        _add_to_dict(all_plot_data, experiment_key, cov_data)

    return all_plot_data


def create_plot_afl_coverage(
    folders: Union[str, pathlib.Path, List[Union[str, pathlib.Path]]], plot_file: str = "plot_data"
):
    """
    Create and return a plot displaying coverage over time extracted from AFL/AFL++ outputs.
    """
    sns.set_theme()
    data_columns = ["edges_found"]
    time_column_aflpp = "# relative_time"
    time_column_afl = "relative_time"
    all_plot_data: Dict[str, pd.DataFrame] = {}

    if not isinstance(folders, list):
        folders = [folders]

    for folder in folders:
        path = pathlib.Path(folder).expanduser()
        try:
            all_plot_data.update(
                _search_afl_plot_data(path, data_columns + [time_column_aflpp], plot_file=plot_file)
            )
        except ValueError:
            all_plot_data.update(
                _search_afl_plot_data(path, data_columns + [time_column_afl], plot_file=plot_file)
            )

    # Preprocess each trial
    for trials in all_plot_data.values():
        for i, trial in enumerate(trials):
            if time_column_afl in trial.columns:
                trial.rename(columns={time_column_afl: time_column_aflpp}, inplace=True)

            idx = np.arange(1, max(86400, int(trial[time_column_aflpp].max())) + 1)
            trial = trial.set_index(time_column_aflpp)
            trial = trial[~trial.index.duplicated()]
            trial = trial.reindex(idx).reset_index()
            trial.ffill(inplace=True)

            if len(idx) > 900:
                trial = trial.loc[trial[time_column_aflpp] % 900 == 0]
            trials[i] = trial

    # Merge trials within experiment
    for exp, trials in all_plot_data.items():
        all_plot_data[exp] = pd.concat(trials, ignore_index=True, sort=False) if len(trials) > 1 else trials[0]

    # Merge experiments
    if len(all_plot_data) > 1:
        plot_data_df = pd.concat(all_plot_data.values(), keys=list(all_plot_data.keys()))
        plot_data_df.reset_index(level=0, inplace=True)
        plot_data_df.reset_index(drop=True, inplace=True)
    else:
        exp, trials = next(iter(all_plot_data.items()))
        plot_data_df = trials
        plot_data_df["level_0"] = exp

    # Rename for plotting
    plot_data_df.rename(
        columns={
            "level_0": "Fuzzer",
            "# relative_time": "Relative time (hours)",
            "edges_found": "Edge coverage",
        },
        inplace=True,
    )

    plot = sns.lineplot(data=plot_data_df, x="Relative time (hours)", y="Edge coverage", hue="Fuzzer")
    plot.legend(loc="lower right")
    plot.set_title("Average edge coverage over time", fontsize=24)
    plot.set_xticks(range(0, 3600 * 25, 3600 * 6))
    plot.set_xticklabels([str(x) for x in range(0, 25, 6)])
    return plot


def _read_last_line_csv(path: pathlib.Path, columns: List[str]) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as plot_file:
        total_cov = pd.read_csv(plot_file, sep=", ", usecols=columns, engine="python").iloc[-1]
    return total_cov


def compute_coverage_experiment(folder: Union[str, pathlib.Path]) -> pd.DataFrame:
    """
    Extract coverage information from an experiment into a Pandas dataframe.
    """
    data_columns = ["edges_found"]
    coverage_info = dict()

    if isinstance(folder, str):
        folder = pathlib.Path(folder).expanduser()

    for target in folder.glob("*"):
        for fuzzer in target.glob("*"):
            total_cov_trials = []
            for trial in fuzzer.glob("trial-*"):
                plot_files = list(trial.glob("**/replayed_plot_data"))
                assert len(plot_files) == 1
                plot_file = plot_files[0]
                total_cov = _read_last_line_csv(plot_file, data_columns)
                total_cov_trials.append(total_cov)

            coverage_info[(target.name, fuzzer.name)] = [
                int(np.mean(total_cov_trials)),
                float(np.std(total_cov_trials)),
            ]

    cov_pd = {
        "index": list(coverage_info.keys()),
        "index_names": ["target", "fuzzer"],
        "columns": ["Avg. edge cov.", "Std. dev."],
        "column_names": ["metrics"],
        "data": list(coverage_info.values()),
    }
    return pd.DataFrame.from_dict(cov_pd, orient="tight")


def replay_corpus(out_path: pathlib.Path, target: pathlib.Path):
    """
    Replay the corpus from `queue` and produce `replayed_plot_data`.
    """
    try:
        subprocess.run(
            [
                str(pathlib.Path("/mlfuzz") / "scripts" / "replay_corpus.py"),
                str(out_path / "queue"),
                str(out_path / "replayed_plot_data"),
                str(target.parent / (target.stem + ".afl")),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as err:
        logger.warning(f"{err}. Skipping corpus replay.")
        print(f"{err}. Skipping corpus replay.")


def kill_fuzzer(fuzzer_command: str = "afl-fuzz", output_stream=subprocess.DEVNULL):
    """
    Kill a fuzzing process by name.
    """
    subprocess.call(["pkill", "-f", fuzzer_command], stdout=output_stream, stderr=output_stream)
