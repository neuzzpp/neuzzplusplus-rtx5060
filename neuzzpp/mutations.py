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
"""This module implements mutation strategies for Neuzz input generation."""
import logging
import pathlib
from typing import Generator, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from neuzzpp.data_loaders import get_seed_len, load_normalized_seeds, read_seeds
from neuzzpp.models import predict_coverage

logger = logging.getLogger(__name__)

# Hardcoded values from AFL++
HAVOC_BLK_SMALL = 32
HAVOC_BLK_MEDIUM = 128
HAVOC_BLK_LARGE = 1500
HAVOC_BLK_XL = 32768
INS_DEL_RATIO = 0.2


def _choose_block_len(limit: int, rng: Optional[np.random.Generator] = None) -> int:
    # Simplify logic here, as we don't have AFL cycles
    rand_ = rng.integers(3) if rng is not None else np.random.randint(3)

    if rand_ == 0:
        min_value = 1
        max_value = HAVOC_BLK_SMALL
    elif rand_ == 1:
        min_value = HAVOC_BLK_SMALL
        max_value = HAVOC_BLK_MEDIUM
    else:
        rand_ = rng.integers(10) if rng is not None else np.random.randint(10)
        if rand_ != 0:
            min_value = HAVOC_BLK_MEDIUM
            max_value = HAVOC_BLK_LARGE
        else:
            min_value = HAVOC_BLK_LARGE
            max_value = HAVOC_BLK_XL

    if min_value >= limit:
        min_value = 1

    rand_limit = min(max_value, limit) - min_value + 1
    rand_val = rng.integers(rand_limit) if rng is not None else np.random.randint(rand_limit)
    return int(min_value + rand_val)


def generate_one_mutation(
    seed: np.ndarray,
    signs: np.ndarray,
    indices: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    n_iter: Optional[int] = None,
) -> Generator[np.ndarray, None, None]:
    """
    Generator producing one by one all mutations for a given seed and gradient info
    targeting a specific edge.

    Args:
        seed: Content of the seed, neither padded nor normalized.
        signs: Signs of gradients for the top or all bytes. Expected values are -1/1.
        indices: Order of top or all bytes in seed as dictated by descending gradient amplitude.
        rng: Numpy random number generator.
        n_iter: Influences the total number of mutations that will be generated. Generate the top
            2^n_iter mutations (roughly).

    Yields:
        Mutated seed.
    """
    len_seed = seed.nbytes
    if n_iter is None:
        n_iter = int(np.log2(len(signs)))
    ins_del_mut_cnt = int(len(signs) * INS_DEL_RATIO)
    iteration_ranges = np.power(2, np.arange(n_iter + 1))  # {0,2,4,8,16,32,...}
    iteration_ranges[0] = 0
    signs_tiled = np.tile(signs, (255, 1))

    for iter_cnt in range(n_iter):
        low_index = int(iteration_ranges[iter_cnt])
        high_index = int(iteration_ranges[iter_cnt + 1])

        # Apply up steps
        up_steps = max(
            [
                255 - seed[indices[index]] if signs[index] == 1 else seed[indices[index]]
                for index in range(low_index, high_index)
            ]
        )
        up_steps = int(up_steps)
        if up_steps > 0:
            mutated_seeds = np.tile(seed, (up_steps, 1))
            mutated_seeds[:, indices[low_index:high_index]] = (
                mutated_seeds[:, indices[low_index:high_index]]
                + (np.arange(1, up_steps + 1) * signs_tiled[:up_steps, low_index:high_index].T).T
            )
            mutated_seeds = np.clip(mutated_seeds, 0, 255)
            for mut in mutated_seeds:
                yield mut
            del mutated_seeds

        # Apply down steps
        down_steps = max(
            [
                255 - seed[indices[index]] if signs[index] == -1 else seed[indices[index]]
                for index in range(low_index, high_index)
            ]
        )
        down_steps = int(down_steps)
        if down_steps > 0:
            mutated_seeds = np.tile(seed, (down_steps, 1))
            mutated_seeds[:, indices[low_index:high_index]] = (
                mutated_seeds[:, indices[low_index:high_index]]
                - (np.arange(1, down_steps + 1) * signs_tiled[:down_steps, low_index:high_index].T).T
            )
            mutated_seeds = np.clip(mutated_seeds, 0, 255)
            for mut in mutated_seeds:
                yield mut
            del mutated_seeds

    while ins_del_mut_cnt > 0:
        ins_del_mut_cnt -= 1
        change_loc = indices[ins_del_mut_cnt // 2]

        # Now do deletions and insertions at top locations
        if ins_del_mut_cnt % 2 == 0:  # Alternating del/ins operations
            cut_len = _choose_block_len(len_seed - int(change_loc), rng=rng)
            mutated_seed = np.concatenate((seed[:change_loc], seed[change_loc + cut_len :]))
            assert len(mutated_seed) == len_seed - cut_len
        else:  # Random insertion operation
            cut_len = _choose_block_len((len_seed - 1) // 2, rng=rng)
            rand_loc = rng.integers(cut_len) if rng is not None else np.random.randint(cut_len)
            mutated_seed = np.concatenate(
                (
                    seed[:change_loc],
                    seed[rand_loc : rand_loc + cut_len],
                    seed[change_loc:],
                )
            )
            assert len(mutated_seed) == len_seed + cut_len
        yield mutated_seed


def generate_all_mutations(
    seed: np.ndarray,
    signs: np.ndarray,
    indices: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    n_iter: Optional[int] = None,
) -> List[np.ndarray]:
    """Generate all mutations for a given seed and target edge."""
    return [mut for mut in generate_one_mutation(seed, signs, indices, rng, n_iter)]


def compute_one_mutation_info(
    model, seed_name: Union[pathlib.Path, str], n_keep_vals: Optional[int] = None
) -> Tuple[List[str], List[str]]:
    """
    Generate and return gradient information for one seed and one mutation targeting
    a random edge.
    """
    # Keras 3 safe: prefer input_shape/output_shape instead of model.input/model.output
    input_dim = int(model.input_shape[-1])
    seed = load_normalized_seeds([seed_name], max_len=input_dim)
    seed_len = min(get_seed_len(seed_name), input_dim)

    if n_keep_vals is None:
        n_keep_vals = seed_len
    else:
        n_keep_vals = min(n_keep_vals, seed_len)

    out_dim = int(model.output_shape[-1])
    target_edge = int(np.random.choice(out_dim))
    sorting_index, gradient = compute_gradient(model, target_edge, seed, n_keep_vals, "sign")
    sorting_index_lst = [str(int(el)) for el in sorting_index]
    gradient_lst = [str(int(el)) for el in gradient]
    return sorting_index_lst, gradient_lst


def compute_all_mutation_info(
    seeds_path: pathlib.Path,
    bitmap: np.ndarray,
    model,
    n_mutations: Optional[int],
    mutation_strategy: str = "sign",
    edge_choice: str = "unseen",
) -> None:
    """Generate and store gradient information for further mutation generation."""
    grads_folder = seeds_path.parent / "grads"
    seed_list = list(seed for seed in seeds_path.glob("*") if seed.is_file())
    input_dim = int(model.input_shape[-1])

    # Determine how many mutations to generate
    if n_mutations is None:
        n_mutations = len(seed_list)
        selected_seeds = seed_list
    else:
        replace = len(seed_list) < n_mutations
        selected_seeds = [
            seed_list[i] for i in np.random.choice(len(seed_list), n_mutations, replace=replace)
        ]
    logger.info(f"Generating {n_mutations} seeds.")

    # Select target edges
    if edge_choice == "rand":
        target_edges = choose_rand_edges(n_mutations, bitmap.shape[-1])
    elif edge_choice == "unseen":
        target_edges = choose_rand_unseen_edges(n_mutations, bitmap)
    else:
        raise ValueError(f"Unknown strategy for target edge choice: {edge_choice}")

    grads_folder.mkdir(parents=True, exist_ok=True)

    for edge, seed_name in zip(target_edges, selected_seeds):
        seed = load_normalized_seeds([seed_name], max_len=input_dim)
        n_keep_vals = min(get_seed_len(seed_name), input_dim)

        sorting_index, gradient = compute_gradient(
            model, int(edge), seed, n_keep_vals, mutation_strategy
        )

        sorting_index_lst = [str(int(el)) for el in sorting_index]
        gradient_lst = [str(int(el)) for el in gradient]
        grad_name = grads_folder / seed_name.name
        with open(grad_name, "a") as f:
            f.write(",".join(sorting_index_lst) + "|" + ",".join(gradient_lst) + "\n")


def choose_rand_edges(n_mutations: int, n_edges: int) -> np.ndarray:
    """Choose edges to be targeted for coverage randomly."""
    return np.random.choice(n_edges, n_mutations, replace=True)


def choose_rand_unseen_edges(n_mutations: int, bitmap: np.ndarray) -> np.ndarray:
    """
    Choose edges to be targeted for coverage among those that were never covered in the bitmap.
    If no uncovered edges are found, random targets are chosen instead.
    """
    unseen_edges = np.where(~bitmap.any(axis=0))[0]

    if len(unseen_edges) == 0:
        logger.info("No uncovered edges found. Picking target edges randomly.")
        return choose_rand_edges(n_mutations, bitmap.shape[-1])

    logger.info("Targeting unseen edges.")
    return np.random.choice(unseen_edges, n_mutations, replace=True)


def compute_gradient(
    model,
    target_edge: int,
    seed: np.ndarray,
    seed_len: int,
    mutation_strategy: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient information to use for mutating the given seed w.r.t. a target edge.
    """
    inputs = tf.cast(seed, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        out = model(inputs)[:, target_edge]
    grads = tf.squeeze(tape.gradient(out, inputs))

    sorting_index = tf.reverse(tf.argsort(tf.abs(grads[:seed_len])), axis=[-1])
    sorting_index_np = sorting_index.numpy()

    if mutation_strategy == "sign":
        val = tf.sign(tf.gather(grads, sorting_index)).numpy().astype(np.int8)
    elif mutation_strategy == "rand":
        val = np.random.choice([1, -1], seed_len, replace=True).astype(np.int8)
    else:
        raise ValueError(f"Unknown mutation strategy: {mutation_strategy}")

    assert len(val) == len(sorting_index_np)
    return sorting_index_np, val


def compute_mutations_success(model: tf.keras.Model, mutations: List[np.ndarray]) -> np.ndarray:
    """Compute predicted success bitmap for given mutations."""
    covered_edges = predict_coverage(model, mutations)
    covered_edges = np.sum(covered_edges, axis=0)
    return covered_edges


def compute_mutations_success_all(
    model: tf.keras.Model,
    seed_list: List[pathlib.Path],
    save_path: Optional[pathlib.Path] = None,
    n_iter: Optional[int] = None,
) -> None:
    """Compute and store success bitmaps for mutations for each seed."""
    n_edges = int(model.output_shape[-1])
    seeds = read_seeds(seed_list)

    save_path.mkdir(parents=True, exist_ok=True)

    for i, seed_file in enumerate(seed_list):
        if (save_path / (seed_file.name + ".npy")).exists():
            logger.info(f"Found mutation file for seed: {seed_file}. Skipping.")
            continue

        grads = parse_grad_file(seed_file.parent.parent / "grads" / seed_file.name)
        n_grads = len(grads)
        if n_grads == 0 or n_grads != n_edges:
            logger.warning(f"Found incomplete gradient file: {seed_file}. Skipping.")
            continue

        summaries = []
        for indices, signs in grads:
            mutations = generate_all_mutations(seeds[i], signs, indices, n_iter=n_iter)
            summaries.append(compute_mutations_success(model, mutations))
            del mutations
        summaries_arr = np.stack(summaries, axis=0)
        del summaries

        with open(str(save_path / seed_file.name) + ".npy", "wb") as f:
            np.save(f, summaries_arr)
        del summaries_arr


def parse_grad_file(path: pathlib.Path) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load and return the content of the gradient file specified as input.

    Format:
        index1,index2,...,indexN|grad1,grad2,...,gradN
    """
    all_grads: List[Tuple[np.ndarray, np.ndarray]] = []
    try:
        with open(path, "r") as grad:
            for line in grad:
                indices, signs = line.split("|")
                indices_arr = np.array(indices.split(","), dtype=int)

                # IMPORTANT: signs are -1/1, NOT boolean.
                signs_arr = np.array(signs.split(","), dtype=int)

                assert len(indices_arr) == len(signs_arr)
                all_grads.append((indices_arr, signs_arr))
    except Exception:
        logger.warning(f"Gradient file not found or unreadable: {path}. Skipping.")

    return all_grads
