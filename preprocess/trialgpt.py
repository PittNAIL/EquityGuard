

"""
TrialGPT-Matching main functions.
"""

import json
from nltk.tokenize import sent_tokenize
import time
import os

# from openai import AzureOpenAI
# import openai

# openai.api_key = os.getenv("OPENAI_API_KEY")

# client = AzureOpenAI(
#     api_version="2023-09-01-preview",
#     azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
#     api_key=os.getenv("OPENAI_API_KEY"),
# )

def parse_criteria(criteria):
    output = ""
    criteria = criteria.split("\n\n")
    
    idx = 0
    for criterion in criteria:
        criterion = criterion.strip()

        if "inclusion criteria" in criterion.lower() or "exclusion criteria" in criterion.lower():
            continue

        if len(criterion) < 5:
            continue
    
        output += f"{idx}. {criterion}\n" 
        idx += 1
    
    return output


def print_trial(
    trial_info: dict,
    inc_exc: str,
) -> str:
    """Given a dict of trial information, returns a string of trial."""
    
    trial = f"Title: {trial_info['brief_title']}\n"
    trial += f"Target diseases: {', '.join(trial_info['diseases_list'])}\n"
    trial += f"Interventions: {', '.join(trial_info['drugs_list'])}\n"
    trial += f"Summary: {trial_info['brief_summary']}\n"
    
    if inc_exc == "inclusion":
        trial += "Inclusion criteria:\n %s\n" % parse_criteria(trial_info['inclusion_criteria'])
    elif inc_exc == "exclusion":
        trial += "Exclusion criteria:\n %s\n" % parse_criteria(trial_info['exclusion_criteria']) 

    return trial


def get_matching_prompt(
    trial_info: dict,
    inc_exc: str,
    patient: str,
) -> str:
    """Output the prompt."""
    prompt = f"You are a helpful assistant for clinical trial recruitment. Your task is to compare a given patient note and the {inc_exc} criteria of a clinical trial to determine the patient's eligibility at the criterion level.\n"

    if inc_exc == "inclusion":
        prompt += "The factors that allow someone to participate in a clinical study are called inclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.\n"
    
    elif inc_exc == "exclusion":
        prompt += "The factors that disqualify someone from participating are called exclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.\n"

    prompt += f"You should check the {inc_exc} criteria one-by-one, and output the following three elements for each criterion:\n"
    prompt += f"\tElement 1. For each {inc_exc} criterion, briefly generate your reasoning process: First, judge whether the criterion is not applicable (not very common), where the patient does not meet the premise of the criterion. Then, check if the patient note contains direct evidence. If so, judge whether the patient meets or does not meet the criterion. If there is no direct evidence, try to infer from existing evidence, and answer one question: If the criterion is true, is it possible that a good patient note will miss such information? If impossible, then you can assume that the criterion is not true. Otherwise, there is not enough information.\n"
    prompt += f"\tElement 2. If there is relevant information, you must generate a list of relevant sentence IDs in the patient note. If there is no relevant information, you must annotate an empty list.\n" 
    prompt += f"\tElement 3. Classify the patient eligibility for this specific {inc_exc} criterion: "
    
    if inc_exc == "inclusion":
        prompt += 'the label must be chosen from {"not applicable", "not enough information", "included", "not included"}. "not applicable" should only be used for criteria that are not applicable to the patient. "not enough information" should be used where the patient note does not contain sufficient information for making the classification. Try to use as less "not enough information" as possible because if the note does not mention a medically important fact, you can assume that the fact is not true for the patient. "included" denotes that the patient meets the inclusion criterion, while "not included" means the reverse.\n'
    elif inc_exc == "exclusion":
        prompt += 'the label must be chosen from {"not applicable", "not enough information", "excluded", "not excluded"}. "not applicable" should only be used for criteria that are not applicable to the patient. "not enough information" should be used where the patient note does not contain sufficient information for making the classification. Try to use as less "not enough information" as possible because if the note does not mention a medically important fact, you can assume that the fact is not true for the patient. "excluded" denotes that the patient meets the exclusion criterion and should be excluded in the trial, while "not excluded" means the reverse.\n'
    
    prompt += "You should output only a JSON dict exactly formatted as: dict{str(criterion_number): list[str(element_1_brief_reasoning), list[int(element_2_sentence_id)], str(element_3_eligibility_label)]}."
    
    user_prompt = f"Here is the patient note, each sentence is led by a sentence_id:\n{patient}\n\n" 
    user_prompt += f"Here is the clinical trial:\n{print_trial(trial_info, inc_exc)}\n\n"
    user_prompt += f"Plain JSON output:"

    return prompt, user_prompt


def trialgpt_matching(trial: dict, patient: str, model: str, client):
    results = {}

    # doing inclusions and exclusions in separate prompts
    for inc_exc in ["inclusion", "exclusion"]:
        system_prompt, user_prompt = get_matching_prompt(trial, inc_exc, patient)
    
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        # model = "gpt-4"
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
#         response.choices[0].message.content
        message = response.choices[0].message.content.strip()
        message = message.strip("`").strip("json")

        try:
            results[inc_exc] = json.loads(message)
        except:
            results[inc_exc] = message

    return results


"""
TrialGPT-Ranking main functions.
"""

import json
from nltk.tokenize import sent_tokenize
import time
import os

# from openai import AzureOpenAI

# client = AzureOpenAI(
#     api_version="2023-09-01-preview",
#     azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
#     api_key=os.getenv("OPENAI_API_KEY"),
# )

def convert_criteria_pred_to_string(
        prediction: dict,
        trial_info: dict,
) -> str:
    """Given the TrialGPT prediction, output the linear string of the criteria."""
    output = ""

    for inc_exc in ["inclusion", "exclusion"]:

        # first get the idx2criterion dict
        idx2criterion = {}
        criteria = trial_info[inc_exc + "_criteria"].split("\n\n")
        
        idx = 0
        for criterion in criteria:
            criterion = criterion.strip()

            if "inclusion criteria" in criterion.lower() or "exclusion criteria" in criterion.lower():
                continue

            if len(criterion) < 5:
                continue
        
            idx2criterion[str(idx)] = criterion
            idx += 1

        for idx, info in enumerate(prediction[inc_exc].items()):
            criterion_idx, preds = info

            if criterion_idx not in idx2criterion:
                continue

            criterion = idx2criterion[criterion_idx]

            if len(preds) != 3:
                continue

            output += f"{inc_exc} criterion {idx}: {criterion}\n"
            output += f"\tPatient relevance: {preds[0]}\n"
            if len(preds[1]) > 0:
                output += f"\tEvident sentences: {preds[1]}\n"
            output += f"\tPatient eligibility: {preds[2]}\n"
    
    return output


def convert_pred_to_prompt(
        patient: str,
        pred: dict,
        trial_info: dict,
) -> str:
    """Convert the prediction to a prompt string."""
    # get the trial string
    trial = f"Title: {trial_info['brief_title']}\n"
    trial += f"Target conditions: {', '.join(trial_info['diseases_list'])}\n"
    trial += f"Summary: {trial_info['brief_summary']}"

    # then get the prediction strings
    pred = convert_criteria_pred_to_string(pred, trial_info)

    # construct the prompt
    prompt = "You are a helpful assistant for clinical trial recruitment. You will be given a patient note, a clinical trial, and the patient eligibility predictions for each criterion.\n"
    prompt += "Your task is to output two scores, a relevance score (R) and an eligibility score (E), between the patient and the clinical trial.\n"
    prompt += "First explain the consideration for determining patient-trial relevance. Predict the relevance score R (0~100), which represents the overall relevance between the patient and the clinical trial. R=0 denotes the patient is totally irrelevant to the clinical trial, and R=100 denotes the patient is exactly relevant to the clinical trial.\n"
    prompt += "Then explain the consideration for determining patient-trial eligibility. Predict the eligibility score E (-R~R), which represents the patient's eligibility to the clinical trial. Note that -R <= E <= R (the absolute value of eligibility cannot be higher than the relevance), where E=-R denotes that the patient is ineligible (not included by any inclusion criteria, or excluded by all exclusion criteria), E=R denotes that the patient is eligible (included by all inclusion criteria, and not excluded by any exclusion criteria), E=0 denotes the patient is neutral (i.e., no relevant information for all inclusion and exclusion criteria).\n"
    prompt += 'Please output a JSON dict formatted as Dict{"relevance_explanation": Str, "relevance_score_R": Float, "eligibility_explanation": Str, "eligibility_score_E": Float}.'


    user_prompt = "Here is the patient note:\n"
    user_prompt += patient + "\n\n"
    user_prompt += "Here is the clinical trial description:\n"
    user_prompt += trial + "\n\n"
    user_prompt += "Here are the criterion-levle eligibility prediction:\n"
    user_prompt += pred + "\n\n"
    user_prompt += "Plain JSON output:"

    return prompt, user_prompt


def trialgpt_aggregation(patient: str, trial_results: dict, trial_info: dict, model: str, client):
    system_prompt, user_prompt = convert_pred_to_prompt(
            patient,
            trial_results,
            trial_info
    )   

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    result = response.choices[0].message.content.strip()
    result = result.strip("`").strip("json")
    result = json.loads(result)

    return result



__author__ = "qiao"

"""
TrialGPT-Ranking main functions.
"""

import json
from nltk.tokenize import sent_tokenize
import time
import os

# from openai import AzureOpenAI

# client = AzureOpenAI(
#     api_version="2023-09-01-preview",
#     azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
#     api_key=os.getenv("OPENAI_API_KEY"),
# )

def convert_criteria_pred_to_string(
        prediction: dict,
        trial_info: dict,
) -> str:
    """Given the TrialGPT prediction, output the linear string of the criteria."""
    output = ""

    for inc_exc in ["inclusion", "exclusion"]:

        # first get the idx2criterion dict
        idx2criterion = {}
        criteria = trial_info[inc_exc + "_criteria"].split("\n\n")
        
        idx = 0
        for criterion in criteria:
            criterion = criterion.strip()

            if "inclusion criteria" in criterion.lower() or "exclusion criteria" in criterion.lower():
                continue

            if len(criterion) < 5:
                continue
        
            idx2criterion[str(idx)] = criterion
            idx += 1

        for idx, info in enumerate(prediction[inc_exc].items()):
            criterion_idx, preds = info

            if criterion_idx not in idx2criterion:
                continue

            criterion = idx2criterion[criterion_idx]

            if len(preds) != 3:
                continue

            output += f"{inc_exc} criterion {idx}: {criterion}\n"
            output += f"\tPatient relevance: {preds[0]}\n"
            if len(preds[1]) > 0:
                output += f"\tEvident sentences: {preds[1]}\n"
            output += f"\tPatient eligibility: {preds[2]}\n"
    
    return output


def convert_pred_to_prompt(
        patient: str,
        pred: dict,
        trial_info: dict,
) -> str:
    """Convert the prediction to a prompt string."""
    # get the trial string
    trial = f"Title: {trial_info['brief_title']}\n"
    trial += f"Target conditions: {', '.join(trial_info['diseases_list'])}\n"
    trial += f"Summary: {trial_info['brief_summary']}"

    # then get the prediction strings
    pred = convert_criteria_pred_to_string(pred, trial_info)

    # construct the prompt
    prompt = "You are a helpful assistant for clinical trial recruitment. You will be given a patient note, a clinical trial, and the patient eligibility predictions for each criterion.\n"
    prompt += "Your task is to output two scores, a relevance score (R) and an eligibility score (E), between the patient and the clinical trial.\n"
    prompt += "First explain the consideration for determining patient-trial relevance. Predict the relevance score R (0~100), which represents the overall relevance between the patient and the clinical trial. R=0 denotes the patient is totally irrelevant to the clinical trial, and R=100 denotes the patient is exactly relevant to the clinical trial.\n"
    prompt += "Then explain the consideration for determining patient-trial eligibility. Predict the eligibility score E (-R~R), which represents the patient's eligibility to the clinical trial. Note that -R <= E <= R (the absolute value of eligibility cannot be higher than the relevance), where E=-R denotes that the patient is ineligible (not included by any inclusion criteria, or excluded by all exclusion criteria), E=R denotes that the patient is eligible (included by all inclusion criteria, and not excluded by any exclusion criteria), E=0 denotes the patient is neutral (i.e., no relevant information for all inclusion and exclusion criteria).\n"
    prompt += 'Please output a JSON dict formatted as Dict{"relevance_explanation": Str, "relevance_score_R": Float, "eligibility_explanation": Str, "eligibility_score_E": Float}.'


    user_prompt = "Here is the patient note:\n"
    user_prompt += patient + "\n\n"
    user_prompt += "Here is the clinical trial description:\n"
    user_prompt += trial + "\n\n"
    user_prompt += "Here are the criterion-levle eligibility prediction:\n"
    user_prompt += pred + "\n\n"
    user_prompt += "Plain JSON output:"

    return prompt, user_prompt


def trialgpt_aggregation(patient: str, trial_results: dict, trial_info: dict, model: str, client):
    system_prompt, user_prompt = convert_pred_to_prompt(
            patient,
            trial_results,
            trial_info
    )   

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    result = response.choices[0].message.content.strip()
    result = result.strip("`").strip("json")
    result = json.loads(result)

    return result