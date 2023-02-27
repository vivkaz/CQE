"""
Perform evaluation on the models, given a test dataset.
"""

import json
import os
import re
from typing import List, Dict

import numpy as np
import pandas as pd

from tagger import NumParserTagger, QuantulumTagger, TextRecognizerTagger, CCG_NILPTagger, tag_entire_file, GPT3Tagger


################################################### Decoding of output ###################################################
def decode_dictionary(result, attribute_list, average_ranges=False, gt_concepts=False):
    ''''
    decode the dictionary we get from the ground truth and the model
    '''

    # values come in different format, if they are strings, they are coming from groud truth and ranges are sepearted with -
    # the model outputs are either a single value or range in form of a tuple
    if isinstance(result["value"], str) and not result["value"].startswith("-") and "-" in result["value"]:
        if average_ranges:
            value = (float(result["value"].split("-")[0]) + float(result["value"].split("-")[1])) / 2
        else:
            value = (float(result["value"].split("-")[0]), float(result["value"].split("-")[1]))
    elif isinstance(result["value"], tuple):
        v1 = result["value"][0]
        v2 = result["value"][1]
        value = (float(v1), float(v2))
    else:
        value = float(result["value"])
    # gt is lowered case so we lower case the unit
    unit = result["normalized_unit"].lower()
    change, referred_concepts = None, None

    # only add changre and concepts if we want to compute them
    if "change" in attribute_list:
        change = result["change"]
    if "referred_concepts" in attribute_list:
        referred_concepts = result["referred_concepts"]
        if gt_concepts:
            referred_concepts = re.split(r'[,\s]+', referred_concepts)  # split with comma and white space
    return value, unit, change, referred_concepts


################################################### Equality checking functions ###################################################

def values_match(gt_value, model_value):
    '''
    do the extracted values match the ground truth?
    '''

    if isinstance(gt_value, tuple) and isinstance(model_value, tuple):
        if np.isclose(gt_value[0], model_value[0], rtol=1e-05, atol=1e-08, equal_nan=False) and np.isclose(gt_value[1],
                                                                                                           model_value[
                                                                                                               1],
                                                                                                           rtol=1e-05,
                                                                                                           atol=1e-08,
                                                                                                           equal_nan=False):
            return True
        else:
            return False
    if isinstance(gt_value, tuple) and not isinstance(model_value, tuple) or not isinstance(gt_value,
                                                                                            tuple) and isinstance(
        model_value, tuple):
        return False
    return np.isclose(gt_value, model_value, rtol=1e-05, atol=1e-08, equal_nan=False)


def concepts_match(gt_concept, model_concept):
    '''
    Do the concepts match, returns partial matches, when we have some of the concepts from the ground truth
    and a compelete match when we have all of the concepts from the ground truth
    '''
    intersection = list(set(gt_concept) & set(model_concept))
    if len(intersection) > 0:
        partial_match = True
    else:
        partial_match = False
    if len(intersection) == len(gt_concept) and len(gt_concept) == len(model_concept):
        compelete_match = True
    else:
        compelete_match = False
    return compelete_match, partial_match


################################################### Calculate Percision, Recall, F1 ###################################################

def get_fscores(true_positive, total_gt, total_model):
    p = true_positive / total_model
    r = true_positive / total_gt
    if p + r == 0:
        return 0, 0, 0
    f1 = (2.0 * p * r) / (p + r)
    return round(p * 100, 1), round(r * 100, 1), round(f1 * 100, 1)


def calculate_metrics(tagging_results: List[Dict], system_name, attribute_list=["value", "unit"],
                      debug=False):  # the quantifier can not normalize
    """
    Given a tagged dataset compute the percision, recall and f1 scores.
    """
    total_tags = 0  # number of quantities extracted by the model (tp+fp)
    total_gt = 0  # number of quantities in the ground truth (tp+fn)
    correct_value = 0  # number
    correct_num_unit = 0  # number + normalized unit
    correct_num_change = 0  # number + normalized unit +change
    correct_concept_compelete = 0  # if the value + concept is correct
    correct_concept_partial = 0  # if the value + concept is partially correct
    average_ranges = False
    if system_name == "quantulum3":
        average_ranges = True
    for result in tagging_results:

        ##### debug
        if debug:
            print(system_name, result["text"])
            print(result["gt"])
            print(result["tags"])
        total_gt = total_gt + len(result["gt"])
        total_tags = total_tags + len(result["tags"])
        tags = result["tags"].copy()
        total_gt_values = len(result["gt"])
        for quantity_gt in result["gt"]:

            gt_value, gt_unit, gt_change, gt_referred_concepts = decode_dictionary(quantity_gt, attribute_list,
                                                                                   gt_concepts=True,
                                                                                   average_ranges=average_ranges)
            for quantity_model in tags:
                model_value, model_unit, model_change, model_referred_concepts = decode_dictionary(quantity_model,
                                                                                                   attribute_list)
                if values_match(gt_value, model_value):
                    if total_gt_values > 0:
                        correct_value = correct_value + 1
                        total_gt_values = total_gt_values - 1  # to avoid counting the same sample more than once
                    if "change" in attribute_list:
                        if model_change == gt_change:
                            correct_num_change = correct_num_change + 1
                        ##### debug
                        elif debug:
                            print("change: ground truth:{}, model:{}".format(gt_change, model_change))
                    if "referred_concepts" in attribute_list:
                        compelete_match, partial_match = concepts_match(gt_referred_concepts, model_referred_concepts)
                        if debug and not compelete_match:
                            print("concept: ground truth:{}, model:{}".format(gt_referred_concepts,
                                                                              model_referred_concepts))
                        if compelete_match: correct_concept_compelete = correct_concept_compelete + 1
                        if partial_match: correct_concept_partial = correct_concept_partial + 1
                    if model_unit == gt_unit:
                        correct_num_unit = correct_num_unit + 1
                    ##### debug
                    elif debug:
                        print("unit: ground truth:{}, model:{}".format(gt_unit, model_unit))
        if debug:  ##### debug
            print("------")
    mertic_dict = []
    precision_value, recall_value, f1_value = get_fscores(correct_value, total_gt, total_tags)
    mertic_dict.append({"P": precision_value, "R": recall_value, "F1": f1_value, "model": system_name, "type": "value"})

    if debug:  ##### debug
        print("correct_num_unit:{},total_gt:{},total_predition:{}".format(correct_num_unit, total_gt, total_tags))
    precision_unit, recall_unit, f1_unit = get_fscores(correct_num_unit, total_gt, total_tags)
    mertic_dict.append(
        {"P": precision_unit, "R": recall_unit, "F1": f1_unit, "model": system_name, "type": "value+unit"})

    if "change" in attribute_list:
        precision_change, recall_change, f1_change = get_fscores(correct_num_change, total_gt, total_tags)
        mertic_dict.append(
            {"P": precision_change, "R": recall_change, "F1": f1_change, "model": system_name, "type": "value+change"})

    if "referred_concepts" in attribute_list:
        precision_partial_concept, recall_partial_concept, f1_partial_concept = get_fscores(correct_concept_partial,
                                                                                            total_gt, total_tags)
        mertic_dict.append({"P": precision_partial_concept,
                            "R": recall_partial_concept, "F1": f1_partial_concept, "model": system_name,
                            "type": "value+concept_partial"})

        precision_compelete_concept, recall_compelete_concept, f1_compelete_concept = get_fscores(
            correct_concept_compelete, total_gt, total_tags)
        mertic_dict.append({"P": precision_compelete_concept,
                            "R": recall_compelete_concept, "F1": f1_compelete_concept, "model": system_name,
                            "type": "value+concept_compelete"})
    if debug: print(mertic_dict)
    return mertic_dict


################################################### Main function ###################################################

def evaluate(test_file, gpt_folder, output_folder):
    ####################### go through each parser #######################
    head, tail = os.path.split(test_file)
    gpt_file = os.path.join(gpt_folder, "predictions_" + tail)
    parser_list = [NumParserTagger(), QuantulumTagger(), TextRecognizerTagger(), CCG_NILPTagger(),
                   GPT3Tagger(gpt_file)]  # NumParserTagger(),QuantulumTagger()
    ####################### read the data the data #######################
    with open(test_file, "r", encoding="utf8") as input_file:
        data = json.load(input_file)
    total_metric_dict = []
    ####################### tag the data #######################
    for parser_ in parser_list:
        print(parser_.name)
        if tail.replace(".json", "") != "NewsQuant":
            attribute_list = ["value", "unit"]  # for these datasets we only have unit and value
            if parser_.name in ["quantulum3", "ccg_nlp", "text-recognizer", "GPT3"]:
                parser_.set_dataset_text_recongizer()
        elif parser_.name == "ccg_nlp":
            attribute_list = ["value", "unit", "change"]
        elif parser_.name == "CQE-Numparser" or parser_.name == "GPT3":
            attribute_list = ["value", "unit", "change", "referred_concepts"]
        else:
            attribute_list = ["value", "unit"]
        tagging_results = tag_entire_file(data, parser_)
        ####################### calculate metrics #######################
        mertic_dict = calculate_metrics(tagging_results, system_name=parser_.name, attribute_list=attribute_list)
        total_metric_dict.extend(mertic_dict)

    ####################### write it as a csv file #######################
    df = pd.DataFrame(total_metric_dict)

    df.to_csv(os.path.join(output_folder, tail.replace(".json", ".csv")), sep='\t')


if __name__ == '__main__':
    ####################### input parameters #######################
    test_file = "../data/formatted_test_set/NewsQuant.json"  # file to test
    gpt_folder = "../data/gpt_3_output/"
    output_folder = "../data/evaluation_output"
    evaluate(test_file, gpt_folder, output_folder)
