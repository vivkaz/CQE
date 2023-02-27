import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from evaluate_models import calculate_metrics
from tagger import NumParserTagger, QuantulumTagger, TextRecognizerTagger, CCG_NILPTagger, tag_entire_file_with_index, \
    GPT3Tagger


def paired_bootstrap_test(gold_labels, system_a_labels, system_b_lables, attribute_list, seed: int = 2023,
                          n_resamples: int = 10_000, ):
    """
    From https://github.com/dennlinger/summaries/blob/main/summaries/evaluation/significance_testing.py
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
    :param n_resamples: Number of times to re-sample the test set.
    :param seed: Random seed to ensure reproducible results.
    :return: p-value of the statistical significance that A is better than B.
    """
    p_value_value, p_value_unit, p_value_change, p_value_concept_partial, p_value_concept_compelete = 0, 0, 0, 0, 0
    system_a_better_value = 0
    system_a_better_unit = 0
    system_a_better_change = 0
    system_a_better_concept_partial = 0
    system_a_better_concept_compelete = 0
    rng = np.random.default_rng(seed=seed)
    # Perform re-sampling with replacement of a similarly large "test set".
    for _ in tqdm(range(n_resamples)):
        indices = rng.choice(len(gold_labels), len(gold_labels), replace=True)
        curr_system_a = [system_a_labels[i] for i in indices.tolist()]
        curr_system_b = [system_b_lables[i] for i in indices.tolist()]

        mertic_dict_a = calculate_metrics(curr_system_a, system_name=sys_a.name, attribute_list=attribute_list,
                                          debug=False)
        mertic_dict_b = calculate_metrics(curr_system_b, system_name=sys_b.name, attribute_list=attribute_list,
                                          debug=False)
        return_dict = decode_metric_dicts(mertic_dict_a, mertic_dict_b, attribute_list)
        if return_dict["value"]["a"] > return_dict["value"]["b"]:
            system_a_better_value += 1
        if return_dict["unit"]["a"] > return_dict["unit"]["b"]:
            system_a_better_unit += 1

        if "change" in return_dict:
            if return_dict["change"]["a"] > return_dict["change"]["b"]:
                system_a_better_change += 1

        if "concept" in return_dict:
            if return_dict["concept_partial"]["a"] > return_dict["concept_partial"]["b"]:
                system_a_better_concept_partial += 1
            if return_dict["concept_compelete"]["a"] > return_dict["concept_compelete"]["b"]:
                system_a_better_concept_compelete += 1

    p_value_value = 1 - (system_a_better_value / float(n_resamples))
    p_value_unit = 1 - (system_a_better_unit / float(n_resamples))

    print(system_a_better_value, system_a_better_unit)

    if "change" in attribute_list:
        p_value_change = 1 - (system_a_better_change / float(n_resamples))
    if "concept" in attribute_list:
        p_value_concept_partial = 1 - (system_a_better_concept_partial / float(n_resamples))
        p_value_concept_compelete = 1 - (system_a_better_concept_compelete / float(n_resamples))
    mertic_dict = {"value": p_value_value, "unit": p_value_unit, "change": p_value_change,
                   "concept_part": p_value_concept_partial,
                   "concept_comp": p_value_concept_compelete}
    return mertic_dict


def decode_metric_dicts(mertic_dict_a, mertic_dict_b, attribute_list):
    """
    Given the dictionary of metrics we extract the F1 values for value, unit and change and concept
    """
    a_value = mertic_dict_a[0]
    b_value = mertic_dict_b[0]
    a_unit = mertic_dict_a[1]
    b_unit = mertic_dict_b[1]
    return_dict = {"value": {"a": a_value["F1"], "b": b_value["F1"]},
                   "unit": {"a": a_unit["F1"], "b": b_unit["F1"]}}
    if "change" in attribute_list:
        a_change = mertic_dict_a[2]
        b_change = mertic_dict_b[2]
        return_dict["change"] = {"a": a_change["F1"], "b": b_change["F1"]}

    if "concept" in attribute_list:
        a_concept_partial = mertic_dict_a[3]
        b_concept_partial = mertic_dict_b[3]
        a_concept_compelete = mertic_dict_a[4]
        b_concept_compelete = mertic_dict_b[4]
        return_dict["concept_partial"] = {"a": a_concept_partial["F1"],
                                          "b": b_concept_partial["F1"]}
        return_dict["concept_compelete"] = {"a": a_concept_compelete["F1"],
                                            "b": b_concept_compelete["F1"]}
    return return_dict


def get_attribute_list(tail, sys_b):
    """
    based on the system and the test file, deciede on the attributes you want to evaluate on
    """
    if tail.replace(".json", "") != "NewsQuant":
        attribute_list = ["value", "unit"]  # for these datasets we only have unit and value
        if sys_b.name in ["quantulum3", "ccg_nlp", "text-recognizer", "GPT3"]:
            sys_b.set_dataset_text_recongizer()
    elif sys_b.name == "ccg_nlp":
        attribute_list = ["value", "unit", "change"]
    elif sys_b.name == "CQE-Numparser" or sys_b.name == "GPT3":
        attribute_list = ["value", "unit", "change", "referred_concepts"]
    else:
        attribute_list = ["value", "unit"]
    return attribute_list


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
        mertic_dict = paired_bootstrap_test(data, system_a_labels, system_b_labels, attribute_list)
        mertic_dict["system_b"] = sys_b.name
        total_metric_dict.append(mertic_dict)
        print("system a:{}, system b:{}".format(sys_a.name, sys_b.name))
        print(mertic_dict)

    df = pd.DataFrame(total_metric_dict)
    df.to_csv(os.path.join(output_folder, tail.replace(".json", "_significance.csv")), sep='\t')
