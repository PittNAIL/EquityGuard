import json
import os
import sys
import pandas as pd
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from openai import OpenAI
import pytrec_eval
from trialgpt import trialgpt_matching, trialgpt_aggregation
from rank_result import get_matching_score, get_agg_score
import argparse

# Load OpenAI API key
def load_openai_key():
    api_key = "Your Openai key" 
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI()
def load_gemini_key():
    import google.generativeai as genai
    api_key ='Your Gemini API key'
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model, generation_config={"response_mime_type": "application/json"})
    return model()

def load_claude_key():
    import anthropic
    client = anthropic.Anthropic(
      # defaults to os.environ.get("ANTHROPIC_API_KEY")
      api_key="Your Claude API key"
    )
    return client()

# Process patient notes
def process_patient(patient):
    sents = sent_tokenize(patient)
    sents.append("The patient will provide informed consent and comply with the trial protocol without any practical issues.")
    sents = [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
    return "\n".join(sents)

# Perform matching for a list of trials and sensitive patient notes
def match_trials(client, model, corpuses, output_dir):
    for corpus in corpuses:
        original_dataset = json.load(open(f"../{corpus}/retrieved_trials.json"))
        social_patient_note = json.load(open(f"sensitive_changed_patient_note/{corpus}_all_sensitive.json"))
        output_path = os.path.join(output_dir, f"matching_results_{corpus}_{model}.json")

        output = load_existing_output(output_path)

        for instance in original_dataset:
            patient_id = instance["patient_id"]
            patient = instance["patient"]
            sensitive_notes = social_patient_note[patient_id]

            processed_patient = process_patient(patient)
            processed_sensitive_notes = {k: process_patient(v) for k, v in sensitive_notes.items() if v}

            if patient_id not in output:
                output[patient_id] = {s_type: {"0": {}, "1": {}, "2": {}} for s_type in processed_sensitive_notes}

            for sensitive_type, sensitive_note in processed_sensitive_notes.items():
                for label in ["2", "1", "0"]:
                    if label not in instance: continue

                    for trial in instance[label]:
                        trial_id = trial["NCTID"]
                        if trial_id in output[patient_id][sensitive_type][label]:
                            continue
                        try:
                            results = trialgpt_matching(trial, sensitive_note, model, client)
                            output[patient_id][sensitive_type][label][trial_id] = results
                            save_output(output, output_path)
                        except Exception as e:
                            print(f"Error in matching: {e}")
                            continue

# Aggregate matching results
def aggregate_results(client, model, corpuses, output_dir):
    for corpus in corpuses:
        queries_file = f"sensitive_changed_patient_note/{corpus}_all_sensitive.json"
        queries = json.load(open(queries_file))
        matching_results_path = os.path.join(output_dir, f"matching_results_{corpus}_{model}.json")
        results = json.load(open(matching_results_path))
        trial2info = json.load(open(f"../{corpus}/trial_info.json"))
        output_path = os.path.join(output_dir, f"aggregation_results_{corpus}_{model}.json")

        output = load_existing_output(output_path)
        sensitive_categories = ["homeless", "low income", "unemployed", "disabled", "gay", "gender", "all", "race", "illiterate", "patient", "lesbien", "transgender"]

        for patient_id, info in tqdm(results.items()):
            if patient_id not in output:
                output[patient_id] = {sens: {} for sens in sensitive_categories}

            for sensitive in sensitive_categories:
                if sensitive not in info:
                    continue
                new_info = info[sensitive]
                patient = process_patient(queries[patient_id][sensitive])

                for label, trials in new_info.items():
                    for trial_id, trial_results in trials.items():
                        if trial_id in output[patient_id]:
                            continue
                        try:
                            trial_info = trial2info[trial_id]
                            result = trialgpt_aggregation(patient, trial_results, trial_info, model, client)
                            output[patient_id][sensitive][trial_id] = result
                            save_output(output, output_path)
                        except Exception as e:
                            print(f"Error in aggregation: {e}")
                            continue

# Rank the results
def rank_results(model, corpuses, output_dir):
    sensitive_categories = ["homeless", "low income", "unemployed", "disabled", "gay", "gender", "all", "race", "illiterate", "patient", "lesbien", "transgender"]
    for corpus in corpuses:
        matching_results_path = os.path.join(output_dir, f"matching_results_{corpus}_{model}.json")
        agg_results_path = os.path.join(output_dir, f"aggregation_results_{corpus}_{model}.json")
        matching_results = json.load(open(matching_results_path))
        agg_results = json.load(open(agg_results_path))

        for sensitive in sensitive_categories:
            data_df = {"query-id": [], "corpus-id": [], "score": []}

            for patient_id, label2trial2results in matching_results.items():
                if sensitive not in label2trial2results:
                    continue

                trial2score = {}
                for _, trial2results in label2trial2results[sensitive].items():
                    for trial_id, results in trial2results.items():
                        try:
                            matching_score = get_matching_score(results)
                            agg_score = get_agg_score(agg_results[patient_id][trial_id]) if trial_id in agg_results[patient_id] else 0
                            trial_score = matching_score + agg_score
                            trial2score[trial_id] = trial_score
                        except Exception as e:
                            continue

                sorted_trial2score = sorted(trial2score.items(), key=lambda x: -x[1])

                for trial, score in sorted_trial2score:
                    data_df["query-id"].append(patient_id)
                    data_df["corpus-id"].append(trial)
                    data_df["score"].append(score)

            df = pd.DataFrame(data_df)
            output_path = os.path.join(output_dir, f"rerank_result/{corpus}/{sensitive}.tsv")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, sep='\t', index=False)

# Calculate NDCG scores
def calculate_ndcg(corpuses, output_dir):
    sensitive_categories = ["homeless", "low income", "unemployed", "disabled", "gay", "gender", "all", "race", "illiterate", "patient", "lesbien", "transgender"]
    for corpus in corpuses:
        qrel_path = f"../{corpus}/qrels/test.tsv"
        qrel_df = pd.read_csv(qrel_path, sep='\t', header=None, names=['query_id', 'doc_id', 'score'])
        qrel = qrel_df.groupby('query_id')['doc_id', 'score'].apply(lambda g: dict(g.values)).to_dict()

        for sensitive in sensitive_categories:
            rerank_path = os.path.join(output_dir, f"rerank_result/{corpus}/{sensitive}.tsv")
            if not os.path.exists(rerank_path):
                continue

            rerank_df = pd.read_csv(rerank_path, sep='\t', header=None, names=['query_id', 'doc_id', 'score'])
            results = rerank_df.groupby('query_id')['doc_id', 'score'].apply(lambda g: dict(g.values)).to_dict()

            evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'ndcg'})
            ndcg_scores = evaluator.evaluate(results)

            ndcg_results = [{'query_id': query_id, 'ndcg': metrics['ndcg']} for query_id, metrics in ndcg_scores.items()]
            ndcg_df = pd.DataFrame(ndcg_results)
            print(f"Sensitive: {sensitive}")
            print(ndcg_df)

# Helper function to load existing output file
def load_existing_output(path):
    return json.load(open(path)) if os.path.exists(path) else {}

# Helper function to save output to file
def save_output(output, path):
    with open(path, "w") as f:
        json.dump(output, f, indent=4)

# Main entry point
def main():
    parser = argparse.ArgumentParser(description="LLM-based Clinical Trial Matching and Aggregation")
    parser.add_argument("--llm_type", type=str, default="gpt-4", help="Type of LLM to use (gpt-4, gemini, claude)")
    args = parser.parse_args()

    model = args.llm_type
    corpuses = ["sigir", "trec_2021", "trec_2022"]
    output_dir = "fairness_results"
    if model == "gpt4":
        client = load_openai_key()
    elif model == "gemini":
        client = load_gemini_key()
    elif model == "claude":
        client = load_claude_key()
        

    match_trials(client, model, corpuses, output_dir)
    aggregate_results(client, model, corpuses, output_dir)
    rank_results(model, corpuses, output_dir)
    calculate_ndcg(corpuses, output_dir)

if __name__ == "__main__":
    main()
