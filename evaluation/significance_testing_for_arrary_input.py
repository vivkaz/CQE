
#taken from https://github.com/dennlinger/summaries/blob/main/summaries/evaluation/significance_testing.py


"""
A "collection" of tests for significance testing.
Special thanks go to Michael Hagmann (https://www.cl.uni-heidelberg.de/statnlpgroup/members/hagmann/)
for fruitful discussions and a gentle introduction to significance testing in NLP.
"""
from typing import List, Callable, Union

import numpy as np


def paired_bootstrap_test(gold_labels: Union[List, np.ndarray],
                          system_a: Union[List, np.ndarray],
                          system_b: Union[List, np.ndarray],
                          scoring_function: Callable,
                          n_resamples: int = 10_000,
                          seed: int = 256) -> float:
    """
    Method to compute a paired bootstrap resampling significance test.
    It will be tested whether system A is significantly better than system B and return the p-value.
    Re-samples a temporary test set of the same size as the original, but with replacements.
    If the score of system A is still better than that of system B, we count it as a "success" and repeat n times.
    This implementation largely follows Philipp Koehn's 2004 work
    "Statistical Significance Tests for Machine Translation Evaluation", see
    https://aclanthology.org/W04-3250/
    :param gold_labels: List of ground truth labels/values for a test set.
    :param system_a: List of predictions of system A on the test set. Assumed to be the "better" system."
    :param system_b: List of predictions of system B on the test set. Assumed to be the "baseline" method.
    :param scoring_function: An arbitrary evaluation function which takes in two lists (system, gold) and produces
        an evaluation score (singular float).
    :param n_resamples: Number of times to re-sample the test set.
    :param seed: Random seed to ensure reproducible results.
    :return: p-value of the statistical significance that A is better than B.
    """

    if len(gold_labels) != len(system_a) != len(system_b):
        raise ValueError("Ensure that gold and system outputs have the same lengths!")

    number_of_times_a_better_b = 0
    equal_scores = 0

    rng = np.random.default_rng(seed=seed)

    # Cast to np.array for easier index access
    gold_labels = np.array(gold_labels)
    system_a = np.array(system_a)
    system_b = np.array(system_b)

    for _ in range(n_resamples):
        # Perform re-sampling with replacement of a similarly large "test set".
        indices = rng.choice(len(gold_labels), len(gold_labels), replace=True)

        curr_gold = gold_labels[indices]
        curr_a = system_a[indices]
        curr_b = system_b[indices]

        # Compute the system evaluation scores under the altered test set
        score_a = scoring_function(curr_a, curr_gold)
        score_b = scoring_function(curr_b, curr_gold)

        # TODO: Investigate whether strict improvements should be counted?
        if score_a > score_b:
            number_of_times_a_better_b += 1

        if score_a == score_b:
            equal_scores += 1

    if equal_scores > 0:
        print(f"Encountered samples in which scores were equal, which are not counted in the returned p-value.\n"
              f"If cases with equal scoring were to be considered a 'win' for system A over B, then the corrected "
              f"p-value would be {1 - ((number_of_times_a_better_b + equal_scores) / n_resamples)}.")

    p_value = 1 - (number_of_times_a_better_b / n_resamples)
    return p_value


def permutation_test(gold_labels: Union[List, np.ndarray],
                     system_a: Union[List, np.ndarray],
                     system_b: Union[List, np.ndarray],
                     scoring_function: Callable,
                     n_resamples: int = 10_000,
                     seed: int = 256) -> float:
    """
    Method to compute a resampling-based significance test.
    It will be tested whether system A is significantly better than system B and return the p-value.
    Permutates the predictions of A and B with 50% likelihood, and scores the altered systems A' and B' again.
    The test counts the number of times the difference between permuted scores is larger than the observed difference
    between A and B, and returns the fraction of times this is the case.
    This implementation follows the algorithm by (Riezler and Maxwell III, 2005), see:
    https://aclanthology.org/W05-0908.pdf
    :param gold_labels: List of ground truth labels/values for a test set.
    :param system_a: List of predictions of system A on the test set. Assumed to be the "better" system."
    :param system_b: List of predictions of system B on the test set. Assumed to be the "baseline" method.
    :param scoring_function: An arbitrary evaluation function which takes in two lists (system, gold) and produces
        an evaluation score (singular float).
    :param n_resamples: Number of times to re-sample the test set.
    :param seed: Random seed to ensure reproducible results.
    :return: p-value of the statistical significance that A is better than B.
    """

    if len(gold_labels) != len(system_a) != len(system_b):
        raise ValueError("Ensure that gold and system outputs have the same lengths!")

    rng = np.random.default_rng(seed=seed)

    # Compute |S_A - S_B|
    base_difference = abs(scoring_function(system_a, gold_labels) - scoring_function(system_b, gold_labels))
    number_times_outliers = 0

    joint_labels = np.array([system_a, system_b])
    gold_labels = np.array(gold_labels)
    for _ in range(n_resamples):
        # Randomly permutes the two lists along an axis
        permutation = rng.permuted(joint_labels, axis=0)

        # Re-build the permuted systems of A' and B'
        permuted_A = permutation[0, :]
        permuted_B = permutation[1, :]

        # Check whether the current hypothesis is a "greater outlier"
        permuted_difference = abs(scoring_function(permuted_A, gold_labels) - scoring_function(permuted_B, gold_labels))
        if permuted_difference >= base_difference:
            number_times_outliers += 1

    # Return the offset p-value, following (Riezler and Maxwell, 2005)
    return (number_times_outliers + 1) / (n_resamples + 1)