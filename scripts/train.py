import argparse
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import AutoTokenizer
from model_structure import UnifiedModel
from torch.utils.data import DataLoader
from ndcg_calculation import ndcg_score  # Ensure you have the NDCG calculation function

# Define Equal Opportunity calculation
def calculate_equal_opportunity(true_labels, predicted_labels, sensitive_attr):
    # True Positive Rate for the specific sensitive group
    tp_rate_sensitive = np.sum((true_labels == 1) & (predicted_labels == 1) & (sensitive_attr == 1)) / np.sum((true_labels == 1) & (sensitive_attr == 1))
    tp_rate_non_sensitive = np.sum((true_labels == 1) & (predicted_labels == 1) & (sensitive_attr == 0)) / np.sum((true_labels == 1) & (sensitive_attr == 0))
    
    eo_difference = abs(tp_rate_sensitive - tp_rate_non_sensitive)
    return eo_difference

# Define Demographic Parity calculation
def calculate_demographic_parity(predicted_labels, sensitive_attr):
    # Positive Rate for different sensitive groups
    positive_rate_sensitive = np.sum((predicted_labels == 1) & (sensitive_attr == 1)) / np.sum(sensitive_attr == 1)
    positive_rate_non_sensitive = np.sum((predicted_labels == 1) & (sensitive_attr == 0)) / np.sum(sensitive_attr == 0)

    dp_difference = abs(positive_rate_sensitive - positive_rate_non_sensitive)
    return dp_difference

# Define Error Rate calculation
def calculate_error_rate(true_labels, predicted_labels):
    return 1 - accuracy_score(true_labels, predicted_labels)

# Inference function
def inference(model, dataloader, device, sensitive_attr_key="sensitive_attr"):
    model.eval()
    all_preds = []
    all_labels = []
    all_sensitive_attrs = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            sensitive_attr = batch[sensitive_attr_key].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, task="qa")
            preds = torch.argmax(outputs, dim=-1).cpu().numpy()
            labels = labels.cpu().numpy()
            sensitive_attr = sensitive_attr.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_sensitive_attrs.extend(sensitive_attr)

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_sensitive_attrs = np.array(all_sensitive_attrs)

    return all_preds, all_labels, all_sensitive_attrs

def main():
    # Command-line arguments
    parser = argparse.ArgumentParser(description="Inference for Equal Opportunity, DP, NDCG, and Error Rate calculations")
    parser.add_argument("--model_name", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference")
    parser.add_argument("--sensitive_attr_key", type=str, default="sensitive_attr", help="Key for sensitive attribute in dataset")
    args = parser.parse_args()

    # Load the model
    device = torch.device(args.device)
    model = UnifiedModel.from_pretrained(args.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load the dataset (make sure your dataloader returns batches with sensitive_attr)
    test_dataset = torch.load(args.data_path)  # Assuming the data is preprocessed and saved
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Inference
    all_preds, all_labels, all_sensitive_attrs = inference(model, dataloader, device, args.sensitive_attr_key)

    # Calculate Equal Opportunity
    eo = calculate_equal_opportunity(all_labels, all_preds, all_sensitive_attrs)
    print(f"Equal Opportunity Difference: {eo}")

    # Calculate Demographic Parity
    dp = calculate_demographic_parity(all_preds, all_sensitive_attrs)
    print(f"Demographic Parity Difference: {dp}")

    # Calculate NDCG Score (assuming you have ground truth relevance scores)
    ndcg = ndcg_score(all_labels, all_preds, k=10)
    print(f"NDCG@10 Score: {ndcg}")

    # Calculate Error Rate
    error_rate = calculate_error_rate(all_labels, all_preds)
    print(f"Error Rate: {error_rate}")

if __name__ == "__main__":
    main()
