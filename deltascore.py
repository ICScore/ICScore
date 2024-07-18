import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
from tqdm import tqdm
import gc
import dotenv
import os
import argparse

dotenv.load_dotenv(dotenv.find_dotenv())

def calculate_sentence_probability(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"].to('cuda')
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits

    condition = sentence.split('!')[0] + '!'
    condition = condition.split('?')[0] + '?'
    condition = condition.split('.')[0] + '.'
    
    condition_inputs = tokenizer(condition, return_tensors="pt")
    condition_input_ids = condition_inputs["input_ids"]
    condition_token_num = condition_input_ids.size(1)
    log_probs = torch.log_softmax(logits, dim=-1)
    
    log_likelihood = 0.0
    token_num = input_ids.size(1) - 1
    for i in range(condition_token_num, token_num):  
        token_id = input_ids[0, i + 1].item()
        token_log_prob = log_probs[0, i, token_id].item()
        log_likelihood += token_log_prob
    p = log_likelihood / (token_num - condition_token_num)
    return p

def main(args):
    args = args.parse_args()
    data_folder = os.listdir(args.data_path)
    df = pd.DataFrame()
    for data in data_folder:
        temp = pd.read_json(f'{args.data_path}/{data}', lines=True)
        temp['file'] = data
        df = pd.concat([df, temp])
    
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    models = [
        "mistralai/Mistral-7B-v0.1",
        "meta-llama/Meta-Llama-3-70B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "google/gemma-7b",
        "mistralai/Mixtral-8x7B-v0.1",
    ]
    
    token = os.getenv('HUGGINGFACE_TOKEN')

    with tqdm(desc='models', total=len(models)) as pbar_m:
        for model_name in models:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                trust_remote_code=True,
                token=token
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

            model.eval()
            model_name = model_name.split('/')[1]
            df[f'{model_name}_score'] = 0
            df[f'{model_name}_transformed_score'] = 0
            df[f'{model_name}_delta_score'] = 0
            print(len(df))
            with tqdm(desc='df', total=len(df)) as pbar_d:
                for i, row in df.iterrows():
                    sentence = row['story']
                    transformed_sentence = row['transformed_story']
                    probability_o = calculate_sentence_probability(sentence, model, tokenizer)
                    probability_p = calculate_sentence_probability(transformed_sentence, model, tokenizer)
                    delta_score = probability_o - probability_p
                    df.at[i, f'{model_name}_score'] = probability_o
                    df.at[i, f'{model_name}_transformed_score'] = probability_p
                    df.at[i, f'{model_name}_delta_score'] = delta_score
                    pbar_d.update(1)
            df.to_json(f'{args.data_path}/{data}', orient='records', lines=True)
            pbar_m.update(1)
            gc.collect()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--data_path', type=str)
    main(args)
    
