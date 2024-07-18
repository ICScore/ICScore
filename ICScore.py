import pandas as pd

ICweight = pd.read_csv('./results/weights.csv')
validation_set = pd.read_json('./datas/scores/validation_set.jsonl', lines=True)

model_list = [  'Meta-Llama-3-70B-Instruct_delta_score',
                'Meta-Llama-3-8B-Instruct_delta_score',
                'Mistral-7B-v0.1_delta_score',
                'Mixtral-8x7B-v0.1_delta_score',
                'gemma-7b_delta_score'
                ]

ICtable = pd.merge(validation_set, ICweight[['transformation', 'average']], on='transformation')

for i in model_list:
    ICtable[f'{i}_ICScore'] = ICtable[i] * ICtable['average']
ICscores = ICtable.groupby('story')[[f'{i}_ICScore' for i in model_list]].sum().to_csv('./results/ICScore.csv')