from CQE.unit_classifier.train_classifier_bert import ambigious_units
from pathlib import Path

TOPDIR = Path(__file__).parent.parent.parent
import spacy
import operator
import os
def get_project_root() -> Path:
    return Path(__file__).parent.parent
#######
#contains code for the disambiguator class, which is combination of bert classifiers
#given a surface form the class returns a correct classifer output using the .cats from the spacy transformer
#models should be located in data/units/unit_models/ for this class to work correctly 
######
class unit_disambiguator():
    def __init__(self):
        self.models = {}
        path = get_project_root()
        for key, values in ambigious_units.items():
            if key in ["C", "B", "P"]:
                file_name = os.path.join(path, "unit_models/train_BIG" + key + ".spacy/model-best")
            elif key=="¥":
                file_name = os.path.join(path, "unit_models/train_yen.spacy/model-best")
            elif key=="′":
                file_name = os.path.join(path, "unit_models/train_ascii'.spacy/model-best")
            elif key=="″":
                file_name = os.path.join(path, "unit_models/train_ascii_doublequote.spacy/model-best")
            elif key=="\"":
                file_name = os.path.join(path, "unit_models/train_doublequote.spacy/model-best")
            else:
                file_name = os.path.join(path, "unit_models/train_" + key + ".spacy/model-best")
            self.models[key] = spacy.load(file_name)

    def disambiguate(self, sentence, surface_form):
        probabilities = self.models[surface_form](sentence).cats
        return max(probabilities.items(), key=operator.itemgetter(1))[0]
