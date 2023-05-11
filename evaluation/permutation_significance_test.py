"""
Permutation testing taken from https://github.com/dennlinger/summaries/blob/main/summaries/evaluation/significance_testing.py
"""

import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from bootstrap_signifcance_test import get_attribute_list, decode_metric_dicts
from evaluate_models import calculate_metrics
from tagger import NumParserTagger, QuantulumTagger, TextRecognizerTagger, CCG_NILPTagger, tag_entire_file_with_index, \
    GPT3Tagger


def permute(permutation, system_a_labels, system_b_lables):
    '''
    Permutes the labels across the shared axis, by switching the output based on the labels of (zero for class a, one for class b)
    permutation:permuted list of classes
    system_a_labels: true labels of system a
    system_b_lables: true labels of system b
    returns: the permuted outputs
    '''
    system_a_labels_shuffled = []
    system_b_labels_shuffled = []
    for flip_prob, label_a, label_b in zip(permutation, system_a_labels,
                                         system_b_lables):  # shows if we should flip or not
        if flip_prob == 0:
            system_a_labels_shuffled.append(label_a)
            system_b_labels_shuffled.append(label_b)
        else:  # then we should flip
            system_a_labels_shuffled.append(label_b)
            system_b_labels_shuffled.append(label_a)
    return system_a_labels_shuffled, system_b_labels_shuffled

def compute_differences(system_a_labels,system_b_lables,sys_a,sys_b,attribute_list):

    mertic_dict_a = calculate_metrics(system_a_labels, system_name=sys_a.name, attribute_list=attribute_list,debug=False)
    mertic_dict_b = calculate_metrics(system_b_lables, system_name=sys_b.name, attribute_list=attribute_list,debug=False)
    decoded_dict = decode_metric_dicts(mertic_dict_a, mertic_dict_b, attribute_list)

    # Compute |S_A - S_B|
    difference_value = abs(decoded_dict["value"]["a"] - decoded_dict["value"]["b"])
    difference_unit = abs(decoded_dict["unit"]["a"] - decoded_dict["unit"]["b"])
    difference_change=None
    difference_concept_partial=None
    difference_concept_compelete=None
    if "change" in attribute_list:
        difference_change = abs(decoded_dict["change"]["a"] - decoded_dict["change"]["b"])
    if "concept" in attribute_list:
        difference_concept_partial = abs(decoded_dict["concept_partial"]["a"] - decoded_dict["concept_partial"]["b"])
        difference_concept_compelete = abs(decoded_dict["concept_compelete"]["a"] - decoded_dict["concept_compelete"]["b"])

    return  difference_value,difference_unit,difference_change,difference_concept_partial,difference_concept_compelete

def permutation_test(system_a_labels, system_b_lables, sys_a, sys_b, attribute_list,
                     n_resamples: int = 10_000,
                     seed: int = 25) :
    """
    Method to compute a resampling-based significance test.
    It will be tested whether system A is significantly better than system B and return the p-value.
    Permutates the predictions of A and B with 50% likelihood, and scores the altered systems A' and B' again.
    The test counts the number of times the difference between permuted scores is larger than the observed difference
    between A and B, and returns the fraction of times this is the case.
    This implementation follows the algorithm by (Riezler and Maxwell III, 2005), see:
    https://aclanthology.org/W05-0908.pdf
    :param system_a: List of predictions of system A on the test set. Assumed to be the "better" system."
    :param system_b: List of predictions of system B on the test set. Assumed to be the "baseline" method.
    :param sys_a: the class for system a
    :param sys_b: the class for system b
    :param attribute_list: list of attributes to be considered
    :param n_resamples: Number of times to re-sample the test set.
    :param seed: Random seed to ensure reproducible results.
    :return: p-value of the statistical significance that A is better than B.
    """
    p_value_value, p_value_unit, p_value_change, p_value_concept_partial, p_value_concept_compelete = 0, 0, 0, 0, 0
    n_outliers_vaule, n_outliers_unit, n_outliers_change, n_outliers_concept_partial, n_outliers_concept_compelet = 0, 0, 0, 0, 0
    rng = np.random.default_rng(seed=seed)
    m = len(system_a_labels)

    if  len(system_a_labels) != len(system_b_labels):
        raise ValueError("Ensure that gold and system outputs have the same lengths!")

    result=compute_differences(system_a_labels,system_b_lables,sys_a,sys_b,attribute_list)
    base_diff_value,base_diff_unit,base_diff_change,base_diff_concept_partial,base_diff_concept_compelete=result

    for _ in tqdm(range(n_resamples)):
        # Randomly permutes the two lists along an axis
        permutation = rng.integers(0, 2, m)
        system_a_labels_shuffled, system_b_labels_shuffled = permute(permutation, system_a_labels, system_b_lables)

        # Check whether the current hypothesis is a "greater outlier"
        result = compute_differences(system_a_labels_shuffled, system_b_labels_shuffled,sys_a,sys_b, attribute_list)
        permuted_diff_value, permuted_diff_unit, permuted_diff_change, permuted_diff_concept_partial, permuted_diff_concept_compelete = result

        if permuted_diff_value >= base_diff_value:
            n_outliers_vaule += 1
        if permuted_diff_unit >= base_diff_unit:
            n_outliers_unit += 1

        ##### if there is change
        if "change" in attribute_list:
            if permuted_diff_change >= base_diff_change:
                n_outliers_change += 1
        ##### if there is concept
        if "concept" in attribute_list:
            if permuted_diff_concept_partial >= base_diff_concept_partial:
                n_outliers_concept_partial += 1
            if permuted_diff_concept_compelete >= base_diff_concept_compelete:
                n_outliers_concept_compelet += 1

    print(n_outliers_vaule, n_outliers_unit, n_outliers_change, n_outliers_concept_partial, n_outliers_concept_compelet)
    p_value_value = (n_outliers_vaule + 1) / (n_resamples + 1)
    p_value_unit = (n_outliers_unit + 1) / (n_resamples + 1)
    if "change" in attribute_list:
        p_value_change = (n_outliers_change + 1) / (n_resamples + 1)
    if "concept" in attribute_list:
        p_value_concept_partial = (n_outliers_concept_partial + 1) / (n_resamples + 1)
        p_value_concept_compelete = (n_outliers_concept_compelet + 1) / (n_resamples + 1)
    # Return the offset p-value, following (Riezler and Maxwell, 2005)
    mertic_dict = {"value": p_value_value, "unit": p_value_unit, "change": p_value_change,
                   "concept_part": p_value_concept_partial,
                   "concept_comp": p_value_concept_compelete}
    return mertic_dict


if __name__ == '__main__':

    test_file = "../data/formatted_test_set/NewsQuant.json"
    gpt_folder = "../data/gpt_3_output/"
    output_folder = "../data/evaluation_output"
    head, tail = os.path.split(test_file)
    gpt_file = os.path.join(gpt_folder, "predictions_" + tail)
    with open(test_file, "r", encoding="utf8") as input_file:
        data = json.load(input_file)
    sys_a = NumParserTagger()
    sys_bs = [QuantulumTagger(), TextRecognizerTagger(), CCG_NILPTagger(), GPT3Tagger(gpt_file)]
    total_metric_dict = []
    system_a_labels = tag_entire_file_with_index(data, sys_a)

    for sys_b in sys_bs:
        print("evaluating against", sys_b.name)
        attribute_list = get_attribute_list(tail, sys_b)
        system_b_labels = tag_entire_file_with_index(data, sys_b)
        mertic_dict = permutation_test( system_a_labels, system_b_labels, sys_a, sys_b, attribute_list)
        mertic_dict["system_b"] = sys_b.name
        total_metric_dict.append(mertic_dict)
        print("system a:{}, system b:{}".format(sys_a.name, sys_b.name))
        print(mertic_dict)

    df = pd.DataFrame(total_metric_dict)
    df.to_csv(os.path.join(output_folder, tail.replace(".json", "_pr_significance.csv")), sep='\t')
