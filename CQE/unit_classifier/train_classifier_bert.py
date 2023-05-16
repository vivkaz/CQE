import json
import spacy
from spacy_download import load_spacy

#before running :
# python -m spacy download en_core_web_trf
#following https://towardsdatascience.com/building-sentiment-classifier-using-spacy-3-0-transformers-c744bfc767b
#to train:
# python -m spacy train config.cfg --output ./data/units/unit_models/train_\".spacy --paths.train ./data/units/train/spacy_train/train_set_\".spacy --paths.dev ./data/units/train/spacy_train/train_set_\".spacy
# python -m spacy train config.cfg --output ./data/units/unit_models/train_\'.spacy --paths.train ./data/units/train/spacy_train/train_set_\'.spacy --paths.dev ./data/units/train/spacy_train/train_set_\'.spacy
# python -m spacy train config.cfg --output ./data/units/unit_models/train_a.spacy --paths.train ./data/units/train/spacy_train/train_set_a.spacy --paths.dev ./data/units/train/spacy_train/train_set_a.spacy
# python -m spacy train config.cfg --output ./data/units/unit_models/train_b.spacy --paths.train ./data/units/train/spacy_train/train_set_b.spacy --paths.dev ./data/units/train/spacy_train/train_set_b.spacy
# python -m spacy train config.cfg --output ./data/units/unit_models/train_c.spacy --paths.train ./data/units/train/spacy_train/train_set_c.spacy --paths.dev ./data/units/train/spacy_train/train_set_c.spacy
# python -m spacy train config.cfg --output ./data/units/unit_models/train_dram.spacy --paths.train ./data/units/train/spacy_train/train_set_dram.spacy --paths.dev ./data/units/train/spacy_train/train_set_dram.spacy
# python -m spacy train config.cfg --output ./data/units/unit_models/train_F.spacy --paths.train ./data/units/train/spacy_train/train_set_F.spacy --paths.dev ./data/units/train/spacy_train/train_set_F.spacy
# python -m spacy train config.cfg --output ./data/units/unit_models/train_kn.spacy --paths.train ./data/units/train/spacy_train/train_set_kn.spacy --paths.dev ./data/units/train/spacy_train/train_set_kn.spacy
# python -m spacy train config.cfg --output ./data/units/unit_models/train_kt.spacy --paths.train ./data/units/train/spacy_train/train_set_kt.spacy --paths.dev ./data/units/train/spacy_train/train_set_kt.spacy
# python -m spacy train config.cfg --output ./data/units/unit_models/train_p.spacy --paths.train ./data/units/train/spacy_train/train_set_p.spacy --paths.dev ./data/units/train/spacy_train/train_set_p.spacy
# python -m spacy train config.cfg --output ./data/units/unit_models/train_pound.spacy --paths.train ./data/units/train/spacy_train/train_set_pound.spacy --paths.dev ./data/units/train/spacy_train/train_set_pound.spacy
# python -m spacy train config.cfg --output ./data/units/unit_models/train_R.spacy --paths.train ./data/units/train/spacy_train/train_set_R.spacy --paths.dev ./data/units/train/spacy_train/train_set_R.spacy
# python -m spacy train config.cfg --output ./data/units/unit_models/train_¥.spacy --paths.train ./data/units/train/spacy_train/train_set_¥.spacy --paths.dev ./data/units/train/spacy_train/train_set_¥.spacy
# python -m spacy train config.cfg --output ./data/units/unit_models/train_′.spacy --paths.train ./data/units/train/spacy_train/train_set_′.spacy --paths.dev ./data/units/train/spacy_train/train_set_′.spacy
# python -m spacy train config.cfg --output ./data/units/unit_models/train_″.spacy --paths.train ./data/units/train/spacy_train/train_set_″.spacy --paths.dev ./data/units/train/spacy_train/train_set_″.spacy
# python -m spacy train config.cfg --output ./data/units/unit_models/train_BIGP.spacy --paths.train ./data/units/train/spacy_train/train_set_BIGP.spacy --paths.dev ./data/units/train/spacy_train/train_set_BIGP.spacy
# python -m spacy train config.cfg --output ./data/units/unit_models/train_BIGC.spacy --paths.train ./data/units/train/spacy_train/train_set_BIGC.spacy --paths.dev ./data/units/train/spacy_train/train_set_BIGC.spacy
# python -m spacy train config.cfg --output ./data/units/unit_models/train_BIGB.spacy --paths.train ./data/units/train/spacy_train/train_set_BIGB.spacy --paths.dev ./data/units/train/spacy_train/train_set_BIGB.spacy


from pathlib import Path
from spacy.tokens import DocBin
TOPDIR = Path(__file__).parent.parent.parent
ambigious_units={'c': ['cent', 'celsius'],
                 '¥': ['chinese yuan', 'japanese yen'],
                 'kn': ['croatian kuna', 'knot'], 'p': ['point', 'penny'],
                 'R': ['south african rand', 'roentgen'], 'b': ['barn', 'bit'], "'": ['foot', 'minute'],
                 '′': ['foot', 'minute'], '"': ['inch', 'second'], '″': ['inch', 'second'], 'C': ['celsius', 'coulomb'],
                 'F': ['fahrenheit', 'farad'], 'kt': ['kiloton', 'knot'], 'B': ['byte', 'bel'],
                 'P': ['poise', 'pixel'],'dram': ['armenian dram', 'dram'], 'pound': ['pound sterling', 'pound-mass'],
                 'a': ['acre', 'year']}

def training_set_per_unit(units,training_data_path):
    training_set_ = []
    path=TOPDIR.joinpath(training_data_path)
    for file in path.iterdir():
        if file.suffix == ".json":
            with file.open("r", encoding="utf-8") as train_file:
                train_json=json.load(train_file)
                for line in train_json:
                    if line["unit"] in units:
                        training_set_.append(line)

    target_names = list(frozenset([i["unit"] for i in training_set_]))
    target_names_dict={}
    for i,name in enumerate(target_names):
        target_names_dict[name]=i

    tuples=[]
    for row in training_set_:
        tuples.append((row["text"],row["unit"]))
    return tuples,target_names_dict

def document(data,target_names_dict):
    #Creating empty list called "text"
    text = []
    nlp=load_spacy("en_core_web_trf")
    for doc, label in nlp.pipe(data, as_tuples = True):
        for name,i in target_names_dict.items():
            if name==label:
                doc.cats[name]=1
            else:
                doc.cats[name]=0
        text.append(doc)

    return(text)



if __name__ == '__main__':
    training_data_path="data/units/train"#locatin of the the raw json file as training
    training_data_output="data/units/train/spacy_train/"#locatin of the outputs from spacy in bin format

    for key,values in ambigious_units.items():
        print("for surface form :",key)
        training_set,target_names_dict = training_set_per_unit(values,training_data_path)
        print("total number of samples",len(training_set))
        print("passing the train dataset into function 'document'....")
        train_docs = document(training_set,target_names_dict)
        print("Creating binary document using DocBin function in spaCy...")
        doc_bin = DocBin(docs = train_docs)
        print("Saving the binary document as train.spacy...." +training_data_output+"train_set_"+key+".spacy")
        if key=="C" or key=="B" or key=="P":
            doc_bin.to_disk(TOPDIR.joinpath(training_data_output+"train_set_BIG"+key+".spacy"))
        else:
            doc_bin.to_disk(TOPDIR.joinpath(+training_data_output+"/train_set_"+key+".spacy"))
        print("=====================")


