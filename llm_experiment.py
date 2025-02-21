# app.py
# -*- coding: utf-8 -*-
"""
This script loads a CSV dataset, processes the data,
loads a quantized Llama model from Hugging Face, and generates
risk assessment completions for login attempts.
"""

import argparse
import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

# ---------------------------
# Check Hugging Face Token
# ---------------------------
def check_token():
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if token is None:
        print("Token do Hub não encontrado! Por favor, defina o token do Hub como uma variável de ambiente.")
        return False
    return token

# ---------------------------
# Data Loading and Processing
# ---------------------------
def load_data(file_path):
    chunksize = 1_000_000
    account_takeover_df = pd.DataFrame()
    not_account_takeover_df = pd.DataFrame()
    USER_IDS = []
    ACCOUNT_TAKEOVER_IDS = []
    
    # Read CSV file in chunks
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunksize)):
        # Extend account takeover user IDs
        ACCOUNT_TAKEOVER_IDS.extend(chunk[chunk["Is Account Takeover"] == True]['User ID'].tolist())
        # Append rows where the user id is in the account takeover list
        account_takeover_df = pd.concat([account_takeover_df, chunk[chunk['User ID'].isin(ACCOUNT_TAKEOVER_IDS)]])
        
        # Gather non-attack user IDs (limit to 141)
        if len(USER_IDS) < 141:
            USER_IDS.extend(chunk[chunk['Is Account Takeover'] == False]['User ID'].unique().tolist())
            if len(USER_IDS) > 141:
                USER_IDS = USER_IDS[:141]
        not_account_takeover_df = pd.concat([not_account_takeover_df, chunk[chunk['User ID'].isin(USER_IDS)]])
    
    return account_takeover_df, not_account_takeover_df, USER_IDS, ACCOUNT_TAKEOVER_IDS

# ---------------------------
# Model Loading
# ---------------------------
def load_model(model_id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)
    return model, tokenizer

# ---------------------------
# Completion Function
# ---------------------------
def get_completion(query, history, model, tokenizer, sector_specific="e-commerce") -> str:
    prompt_template = """
<s>
[INST]
You are an expert cybersecurity analyst. Below is a log line containing data from a login attempt made by a real user of a {sector_specific} website. You also have
a list of the recent attempts made by the user to the same website.
Write an appropriate response in the specified format that will be used to assess the risk of that login attempt in respect to the user and the store. Do not add any
additional messages that is not in the format below.

Format:
{format}

Log data:
{history}
[/INST]
</s>

<s> <|system|>
You are an AI assistant specialised in assessing risks of authentication attempts on a website.</s>

<s> <|user|>
Current attempt login data:
{query}
</s>
"""
    format_str = """
Risk: float, range 0 to 1. Risk of the login attempt being malicious.
Confidence: float, range 0 to 1. Your confidence on the assessment.
Reason: string, short reason of why you attributed these scores.
"""
    prompt = prompt_template.format(format=format_str, history=history, query=query, sector_specific=sector_specific)
    encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_inputs = encodeds.to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1000,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Skip the tokens that correspond to the prompt
    prompt_length = model_inputs["input_ids"].shape[-1]
    generated_tokens = generated_ids[0][prompt_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text

# ---------------------------
# Main Execution Function
# ---------------------------
def main():
    # Check for flag --dataset to specify the dataset file
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Path to the dataset file")
    args = parser.parse_args()

    # Set the path to your CSV dataset
    data_file = "rba-dataset.csv"
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found!")
        return
    
    
    if args.dataset:
        data_file = args.dataset
    
    print("Loading data...")
    account_takeover_df, not_account_takeover_df, USER_IDS, ACCOUNT_TAKEOVER_IDS = load_data(data_file)
    print("Account takeover shape:", account_takeover_df.shape)
    print("Not account takeover shape:", not_account_takeover_df.shape)
    
    # Save both datasets as one CSV file for reference
    combined_df = pd.concat([account_takeover_df, not_account_takeover_df])
    combined_df.to_csv("combined_data.csv", index=False)

    # Login to Hugging Face if necessary (for private models or rate limits).
    login(token=os.environ.get("HUGGINGFACE_TOKEN"))

    # Load the model and tokenizer
    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    print(f"Loading model {model_id} ...")
    model, tokenizer = load_model(model_id)
    
    results = []
    
    # Process logs for account takeover user IDs (limit to 40)
    print("Generating completions for account takeover logs...")
    # Ensure we only consider the rows with account takeover IDs
    account_takeover_df = account_takeover_df[account_takeover_df['User ID'].isin(ACCOUNT_TAKEOVER_IDS)]
    for i, user_id in enumerate(ACCOUNT_TAKEOVER_IDS):
        # Filter and sort logs for the current user
        user_logs = account_takeover_df[account_takeover_df['User ID'] == user_id].sort_values(by='Login Timestamp', ascending=False).head(10)
        user_logs.reset_index(drop=True, inplace=True)
        # Drop unwanted columns if present
        for col in ['index', 'Is Account Takeover']:
            if col in user_logs.columns:
                user_logs.drop(columns=[col], inplace=True)
        
        if user_logs.empty:
            continue

        latest_log = user_logs.iloc[0].to_dict()
        remaining_logs = user_logs.iloc[1:].to_dict()
        
        print(f"Generating completion for account takeover User ID: {user_id}")
        torch.cuda.empty_cache()
        completion = get_completion(query=str(latest_log), history=str(remaining_logs), model=model, tokenizer=tokenizer)
        results.append({'User ID': user_id, 'Resultado': completion})
        print("Resultado:", completion)
    
    # Process logs for non-account takeover user IDs (limit to 40)
    print("Generating completions for non-account takeover logs...")
    for i, user_id in enumerate(USER_IDS):
        user_logs = not_account_takeover_df[not_account_takeover_df['User ID'] == user_id].sort_values(by='Login Timestamp', ascending=False).head(10)
        user_logs.reset_index(drop=True, inplace=True)
        for col in ['index', 'Is Account Takeover']:
            if col in user_logs.columns:
                user_logs.drop(columns=[col], inplace=True)
        
        if user_logs.empty:
            continue

        latest_log = user_logs.iloc[0].to_dict()
        remaining_logs = user_logs.iloc[1:].to_dict()
        
        print(f"Generating completion for non-account takeover User ID: {user_id}")
        torch.cuda.empty_cache()
        completion = get_completion(query=str(latest_log), history=str(remaining_logs), model=model, tokenizer=tokenizer)
        results.append({'User ID': user_id, 'Resultado': completion})
        print("Resultado:", completion)
    
    # Determine the output to /ws/results
    output_dir = "/ws/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to a JSON file
    output_file = os.path.join(output_dir, f"results_{model_id}.json")
    with open(output_file, 'w') as fout:
        json.dump(results, fout)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    hf_token = check_token()
    if not hf_token:
        exit(1)
    main()
