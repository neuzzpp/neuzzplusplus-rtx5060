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
"""Module containing data preprocessing functions."""
import logging
import os
import pathlib
import subprocess
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CoverageBuilder:
    """
    Helper class for creating coverage labels for all seeds in a folder.
    """

    def __init__(self, target: List[str]) -> None:
        """
        Args:
            target: Target programs with args in a callable form.
        """
        afl_path = os.environ.get("AFL_PATH", ".")
        self.command = [
            os.path.join(afl_path, "afl-showmap"),
            "-qe",
            "-o",
            "/dev/stdout",
            "-t",
            "10000",
            "-m",
            "none",
        ] + target

        # Find position in command where seed should be
        try:
            self._seed_position = self.command.index("@@")
        except ValueError:
            self.command.append("@@")
            self._seed_position = len(self.command) - 1

    def get_command_for_seed(self, seed: pathlib.Path) -> List[str]:
        """
        Generate the command to call for extracting the desired type of coverage
        for the seed provided as input.
        """
        self.command[self._seed_position] = str(seed)
        return self.command


def create_path_coverage_bitmap(
    target_with_args: List[str],
    seed_list: List[pathlib.Path],
) -> Dict[pathlib.Path, Optional[Set[int]]]:
    """
    Create edge coverage bitmaps for each seed in `seed_list`.
    """
    logger.info("Creating edge coverage bitmaps.")
    raw_bitmap: Dict[pathlib.Path, Optional[Set[int]]] = {}
    cov_tool = CoverageBuilder(target_with_args)

    has_failed_seeds = False
    for seed in seed_list:
        try:
            command = cov_tool.get_command_for_seed(seed)

            # Use subprocess.run for better control (timeout, stderr capture)
            # NOTE: afl-showmap has its own -t timeout, but this guards against hangs anyway.
            proc = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=30,  # conservative guard; adjust if needed
            )

            edges_curr_seed: Set[int] = set()
            for line in proc.stdout.splitlines():
                # format: b"12345:1" -> take left side
                edge = int(line.split(b":", 1)[0])
                edges_curr_seed.add(edge)

            raw_bitmap[seed] = edges_curr_seed

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as err:
            raw_bitmap[seed] = None
            has_failed_seeds = True
            logger.error(f"Bitmap extraction failed for {seed}: {err}")

    if has_failed_seeds:
        seed_list, raw_bitmap = _clean_seed_list(seed_list, raw_bitmap)

    if not seed_list:
        raise ValueError("No valid seed labels were produced. Stopping.")

    return raw_bitmap


def create_bitmap_from_raw_coverage(
    raw_bitmap: Dict[pathlib.Path, Set[int]]
) -> Tuple[List[pathlib.Path], np.ndarray]:
    """
    Given raw coverage information for seeds, create a compact and compressed
    coverage bitmap.
    """
    seed_list = list(raw_bitmap.keys())
    if not seed_list:
        raise ValueError("Empty raw bitmap: no seeds to process.")

    # Robust union (safe even if some sets are empty)
    all_edges: Set[int] = set()
    for s in raw_bitmap.values():
        all_edges |= set(s)

    if not all_edges:
        # All seeds produced empty coverage -> bitmap is all zeros, keep shape stable
        cov_bitmap = np.zeros((len(seed_list), 0), dtype=bool)
        reduced_bitmap = cov_bitmap
        logger.info(f"Bitmap reduced from {str(cov_bitmap.shape)} to {str(reduced_bitmap.shape)}")
        return seed_list, reduced_bitmap

    all_edges_indices = {addr: index for index, addr in enumerate(all_edges)}
    cov_bitmap = np.zeros((len(seed_list), len(all_edges)), dtype=bool)

    for seed_idx, seed in enumerate(seed_list):
        for addr in raw_bitmap[seed]:
            cov_bitmap[seed_idx][all_edges_indices[addr]] = True

    reduced_bitmap = remove_identical_coverage(cov_bitmap)
    assert len(seed_list) == reduced_bitmap.shape[0]
    return seed_list, reduced_bitmap


def remove_identical_coverage(bitmap: np.ndarray, keep_unseen: bool = False) -> np.ndarray:
    """
    Reduce a coverage bitmap by merging together all edge coverage with identical
    coverage under all seeds.
    """
    if keep_unseen:
        mask = np.sum(bitmap, axis=0) == 0
        _, ind_unique_cov = np.unique(bitmap, axis=1, return_index=True)
        mask[ind_unique_cov] = 1
        reduced_bitmap = bitmap[:, mask]
    else:
        reduced_bitmap = np.unique(bitmap, axis=1)

    logger.info(f"Bitmap reduced from {str(bitmap.shape)} to {str(reduced_bitmap.shape)}")
    return reduced_bitmap


def _clean_seed_list(
    seed_list: List[pathlib.Path],
    raw_bitmap: Dict[pathlib.Path, Optional[Set[int]]],
) -> Tuple[List[pathlib.Path], Dict[pathlib.Path, Set[int]]]:
    logger.info("Removing failed seeds from dataset.")
    cleaned: List[pathlib.Path] = []
    n_removed = 0

    for seed in seed_list:
        if raw_bitmap.get(seed) is None:
            n_removed += 1
        else:
            cleaned.append(seed)

    # Remove failed from dict, keep only good ones
    raw_bitmap_clean: Dict[pathlib.Path, Set[int]] = {
        k: v for k, v in raw_bitmap.items() if v is not None
    }

    logger.info(f"Successfully removed {n_removed} seeds.")
    assert len(cleaned) == len(raw_bitmap_clean), (
        f"The number of seeds and labels do not match: {len(cleaned)}, {len(raw_bitmap_clean)}"
    )
    return cleaned, raw_bitmap_clean
