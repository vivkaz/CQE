from pathlib import Path

TOPDIR = Path(__file__).parent.parent
import json
from numparser.unit_classifier.unit_disambiguator import unit_disambiguator
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from tqdm import tqdm
from quantulum3 import parser as quantparsesr
from quantulum3 import classifier
from significance_testing_for_arrary_input import permutation_test


def custom_f1_score(gt, preds):
    """
    micro averaged f1 score, to account for different class weights
    """

    return f1_score(gt, preds, average="micro")


def test_set(test_path):
    """
    returns the entire test set
    """
    classifier.train_classifier()

    training_set_ = []
    path = TOPDIR.joinpath(test_path)
    for file in path.iterdir():
        if file.suffix == ".json":
            with file.open("r", encoding="utf-8") as train_file:
                train_json = json.load(train_file)
                for line in train_json:
                    training_set_.append(line)

    target_names = list(frozenset([i["unit"] for i in training_set_]))
    target_names_dict = {}
    for i, name in enumerate(target_names):
        target_names_dict[name] = i

    tuples = []
    for row in training_set_:
        tuples.append((row["text"], row["surface_form"], target_names_dict[row["unit"]]))
    return tuples, target_names_dict


def clean_quantlum(p):
    unit_quntulum = p.unit.name
    if p.unit.name == "japanese yen ampere-turn":
        unit_quntulum = "japanese yen"
    if p.unit.name == "minute of arc":
        unit_quntulum = "minute"
    if p.unit.name == "second of arc":
        unit_quntulum = "second"
    if p.unit.name == "degree fahrenheit":
        unit_quntulum = "fahrenheit"
    if p.unit.name == "degree Celsius":
        unit_quntulum = "celsiust"
    if p.unit.name == "megabyte":
        unit_quntulum = "byte"
    if p.unit.name == "decibel":
        unit_quntulum = "bel"
    if p.unit.name == "megabyte per second":
        unit_quntulum = "byte"
    if p.unit.name == "decibel per watt":
        unit_quntulum = "bel"
    if p.unit.name == "megapixel":
        unit_quntulum = "pixel"
    if p.unit.name == "pixel per inch":
        unit_quntulum = "pixel"
    if p.unit.name == "pound-mass ampere-turn":
        unit_quntulum = "pound-mass"
    if p.unit.name == "dime roentgen attometre":
        unit_quntulum = "roentgen"
    if p.unit.name == "second per second":
        unit_quntulum = "roentgen"
    if p.unit.name == "megabit per second":
        unit_quntulum = "bit"
    if p.unit.name == "gigabyte":
        unit_quntulum = "byte"
    if p.unit.name == "celsiust":
        unit_quntulum = "byte"
    if p.unit.name == "chinese yuan ampere-turn":
        unit_quntulum = "chinese yuan"
    if p.unit.name == "south african rand per litre":
        unit_quntulum = "south african rand"
    if p.unit.name == "knot per hour":
        unit_quntulum = "knot"
    if p.unit.name == "roentgen per day":
        unit_quntulum = "roentgen"
    if p.unit.name == "minute of arc to the 6 inch":
        unit_quntulum = "minute"
    if p.unit.name == "picobarn":
        unit_quntulum = "barn"
    if p.unit.name == "knot ampere-turn":
        unit_quntulum = "knot"
    if p.unit.name == "decibel metre":
        unit_quntulum = "decibel"
    if p.unit.name == "second of arc to the 30 minute":
        unit_quntulum = "second"
    return unit_quntulum


if __name__ == '__main__':
    testset, target_names_dict = test_set("data/units/test/")
    gt = []
    preds = []
    quantulum_preds = []
    for text, surface_form, unit in tqdm(testset):
        dis = unit_disambiguator()
        gt.append(unit)
        prediction = dis.disambiguate(text, surface_form)
        prediction_quntulum = quantparsesr.parse(text)
        preds.append(target_names_dict[prediction])

        found = False
        for p in prediction_quntulum:
            unit_quntulum = clean_quantlum(p)
            if unit_quntulum in target_names_dict:
                quantulum_preds.append(target_names_dict[unit_quntulum])
                found = True
                break;

        if not found:
            quantulum_preds.append(0)
    print("num parser:", f1_score(gt, preds, average="micro"))
    print("quantulum:", f1_score(gt, quantulum_preds, average="micro"))
    lables = list(target_names_dict.values())
    target_names = list(target_names_dict.keys())
    report = classification_report(gt, preds, labels=lables, target_names=target_names, zero_division=0, digits=4)
    print(report)
    report = classification_report(gt, quantulum_preds, labels=lables, target_names=target_names, zero_division=0,
                                   digits=4)
    print(report)

    print("p-value:", permutation_test(gt, preds, quantulum_preds, custom_f1_score))
