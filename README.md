# EquityGuard: Enhancing Equity in Large Language Models for Medical Applications

## Overview

**EquityGuard** is a contrastive learning-based framework designed to detect and mitigate biases in Large Language Models (LLMs) used in healthcare applications. The framework addresses inequities observed in tasks such as clinical trial matching (CTM) and medical question answering (MQA), which are crucial for clinical decision support and translational research. By systematically disentangling sensitive attributes such as race, sex, and social determinants of health (SDOH), EquityGuard promotes fairer and more equitable healthcare outcomes.

## Key Features

- **Bias Detection Mechanism**: Identifies and corrects unfair predictions in LLM-based systems.
- **Contrastive Learning**: Uses self-supervised techniques to align data representations, mitigating inequity by targeting biased inputs.
- **Task-Specific Implementation**: Applied to clinical trial matching and medical question-answering tasks while maintaining high performance and fairness.
- **Extensive Evaluation**: Assessed on SIGIR, TREC 2021, TREC 2022, MedQA, and MedMCQA using models like GPT-4, Gemini, and Claude.

## Installation

To use EquityGuard, clone the repository and install the required dependencies:

```bash
git clone https://github.com/PittNAIL/EquityGuard.git
cd EquityGuard
pip install -r requirements.txt
```

## Tasks

### Clinical Trial Matching

EquityGuard automates matching patients to appropriate clinical trials based on eligibility criteria from patient records and trial protocols. It minimizes bias related to race, gender, and other SDOH factors, ensuring equitable recruitment for clinical trials.

### Medical Question Answering (MedQA)

EquityGuard addresses inequities in LLMs used for medical question answering (Q&A), ensuring fair responses across sensitive categories. By mitigating biases, the framework improves the accuracy and fairness of answers provided by LLMs in clinical decision support systems.

## Datasets

The framework was tested on the following datasets:

- **SIGIR 2016**: Clinical trial descriptions from ClinicalTrials.gov and patient case reports.
- **TREC 2021 and 2022**: Datasets focusing on automating the clinical trial matching process.
- **MedQA**: A large-scale dataset containing medical questions from the Chinese medical licensing exam.
- **MedMCQA**: A multi-choice question-answering dataset based on medical topics from AIIMS and NEET PG exams.

For the different tasks we use the different process methods.
1. CTM task
```
python preprocess/change_question_cmt.py --llm_type gpt4 
```

2. MQA task

```
python preprocess/change_question_mqa.py --llm_type gpt4
```

The llm_type is the different llms' name. We have option `gpt4`, `claude` and `gemini`.


## Usage


### Train

The framework can be applied to both clinical trial matching and medical question answering tasks. Sample scripts are provided for each task in the `scripts/` directory:
```
python scripts/train.py --model_name llama3_8B --task qa --epochs 5 --batch_size 16 --lr 1e-5

```

The model name should be `llama3_8B` or `Mistralv0.3`


### Inference

```
python scripts/inference.py --model_name llama3_8B --data_path ./data/test_data.pth --batch_size 16 --device cuda --sensitive_attr_key sensitive_attr
```


- **Equal Opportunity (EO) Calculation**: It computes the True Positive Rate (TPR) for both the sensitive group and non-sensitive group and calculates the absolute difference.
- **Demographic Parity (DP) Calculation**: It calculates the Positive Rate for both groups (sensitive and non-sensitive) and returns the absolute difference between the two.
- **Error Rate Calculation**: The error rate is simply calculated as 1 minus the accuracy score.
- **Inference Function**: The model processes batches of input data, and for each batch, predictions are made. Sensitive attributes are used to calculate fairness metrics (EO and DP).

## Citation
> Ji, Yuelyu, Wenhe Ma, Sonish Sivarajkumar, Hang Zhang, Eugene Mathew Sadhu, Zhuochun Li, Xizhi Wu, Shyam Visweswaran, and Yanshan Wang. "*Mitigating the Risk of Health Inequity Exacerbated by Large Language Models.*" NPJ Digital Medicine. 2025.
