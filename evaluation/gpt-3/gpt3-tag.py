
"""
Uses a long context prediction setting for GPT-3.
"""

import json
from typing import List, Tuple, Dict
import regex

from tqdm import tqdm
import openai

def clean_predictions(text: str) -> List[str]:
    """
    Post-processing of files, by trying different strategies to coerce it into actual singular predictions.
    :param text: Unfiltered text predicted by a language model
    :return: List of individual predictions
    """
    print("original text:",text)
    text=text.replace("0,000","0000")# a common mistake
    # Clear additional clutter that might have been encountered
    text = text.strip(":;.?!")

    # Presence of newlines within the prediction indicates prediction as list
    if "\n" in text:
        cleaned_predictions = text.split("\n")

    # # Other common format contained comma-separated list without anything else
    # elif "," in text.strip("\n "):
    #     cleaned_predictions = [pred.strip(" ") for pred in text.strip("\n ").split(",")]

    else:
        return []


    # # Remove numerals
    cleaned_predictions = [remove_numerals(pred) for pred in cleaned_predictions]
    print("cleaned prediction",str(cleaned_predictions))
    # Make sure everything is lower-cased and stripped
    cleaned_predictions = [pred.strip(" \n") for pred in cleaned_predictions]
    # Remove empty predictions that may have slipped through:
    cleaned_predictions = remove_empty_predictions(cleaned_predictions)

    quants= [convert_to_numbers(pred) for pred in cleaned_predictions]

    return quants
def convert_to_numbers(text:str)->List:
    """
    Takes a prediction and turns it into a list of attributes
    :param text: cleaned text
    :return: list
    """
    parts= text.split(",")
    if len(parts)<5:
        print("does not have enough parts",str(parts))
        return []
    return parts
def remove_numerals(text: str) -> str:
    """
    Will remove any leading numerals (optionally with a dot).
    :param text: Input text, potentially containing a leading numeral
    :return: cleaned text
    """
    if text.startswith("1.") or text.startswith("2.")or text.startswith("3.") or text.startswith("4.") or text.startswith("5.") or text.startswith("6.") or text.startswith("7.") or text.startswith("8.") or text.startswith("9."):
        return text[2:]
    else:
        return text

def remove_empty_predictions(predictions: List[str]) -> List[str]:
    return [pred for pred in predictions if pred.strip("\n ")]

def get_prompts_and_temperatures(sentence) -> List[Tuple[str, str, float]]:


    few_shot_prompt = f"Tag quantities and units in the texts: \n" \
                      f"Sentence: Woot is selling refurbished, unlocked iPhone XR phones with 64GB of storage for about $330. \n" \
                      f"Answer:\n" \
                      f"1. =,1.64, GB, gigabyte, storage  \n2. ~ ,330, $, dollar, iPhone XR phones\n\n" \
                      f"The chain operates more than 600 supermarkets and less than 800 convenience stores.  \n" \
                      f"Answer:\n" \
                      f"1. >, 600, supermarkets, supermarkets, chain  \n2. < ,800, convenience stores, convenience stores, chain \n\n" \
                      f"The spacecraft, which is about the size of a school bus, flew into Dimorphos at a speed of about 4.1 miles per second, that's roughly 14,760 miles per hour (23,760 kilometers per hour).\n" \
                      f"Answer:\n" \
                      f"1. ~ ,1.4.1 , miles per second, miles / second, spacecraft \n2. ~,14760, miles per hour, miles / hour, spacecraft\n 3. ~ ,23760, kilometers per hour, kilometers / hour, spacecraft \n\n" \
                      f"And overnight dogecoin fell from 0.317 to 0.308, a 2.8 percent drop.  \n" \
                      f"Answer:\n" \
                      f"1. = ,1.0.317-0.308 , -, -, dogecoin \n2. =, 2.8, percent, percentage, dogecoin \n\n" \
                      f"This is about minus 387 Fahrenheit (minus 233 Celsius).   \n" \
                      f"Answer:\n" \
                      f"1.~, -387, Fahrenheit, Fahrenheit, - \n2.~, -233, Celsius, Celsius, - \n\n" \
                      f"WhatsApps more than 2 billion users send fewer than 100bn messages a day.  \n" \
                      f"Answer:\n" \
                      f"1. >, 2000000000, users, users, WhatsApp \n2. < ,100000000000, messages, messages, users \n\n" \
                      f"This includes colours between red and blue - wavelengths ranging between 390 and 700 nm.  \n" \
                      f"Answer:\n" \
                      f"1. =, 390-700, nm, nanometer, wavelengths  \n\n" \
                      f"You dont have a two-year bachelors degree or a six to eight year phd degree.   \n" \
                      f"Answer:\n" \
                      f"1. =, 2, year, year, bachelors degree \n2. = ,6-8, year, year, phd degree \n\n" \
                      f" The price CO2 and fuel consumption is not clear.  \n" \
                      f"Answer:\n" \
                      f"No quantities or units\n\n" \
                      f"Sentence:{sentence}\n" \
                      f"Answer:"



    return few_shot_prompt, 0.5


if __name__ == '__main__':
    continue_from = 0 # sometimes it stops in the middle and you need to resume
    input_file= 'input/finance500-ground-truth.json'
    output_file= "output/predictions_finance500.json"
    text_field="text"#Input or text based on the groundtruth file.

    with open(input_file) as f:
        lines = json.load(f)


    openai.api_key = "YOUR_AP_KEY"

    ensemble_predictions = []


    for idx, line in enumerate(tqdm(lines)):
        # Skip already processed samples
        if idx < continue_from:
            continue

        aggregated_predictions = []

        # Extract context and complex word
        sentence = line[text_field]

        # Get "ensemble prompts"
        prompt, temperature = get_prompts_and_temperatures(sentence)

        # Have not experimented too much with other parameters, but these generally worked well.
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            stream=False,
            temperature=temperature,
            max_tokens=512,
            top_p=1,
            best_of=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        predictions = response["choices"][0]["text"]
        print(sentence)
        quants = clean_predictions(predictions)
        print(quants)
        print("--------")
        ensemble_predictions.append({"text":sentence,"quantities":quants})

        with open(output_file, "w") as f:
            json.dump(ensemble_predictions, f, ensure_ascii=False, indent=2)

