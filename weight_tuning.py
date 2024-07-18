import pandas as pd
weight_tuning = pd.read_json('./datas/scores/weight_tuning_set.jsonl', lines=True)

model_list = ['Meta-Llama-3-70B-Instruct_delta_score',
              'Meta-Llama-3-8B-Instruct_delta_score',
              'Mistral-7B-v0.1_delta_score',
              'Mixtral-8x7B-v0.1_delta_score',
              'gemma-7b_delta_score'
                ]

print(weight_tuning.columns)

correlation_i = weight_tuning.groupby('transformation').apply(lambda x: x[model_list].corrwith(-x['Total Interestingness Average'], method='kendall'))
correlation_c = weight_tuning.groupby('transformation').apply(lambda x: x[model_list].corrwith(-x['Total Creativity Average'], method='kendall'))

correlation_i['mean'] = correlation_i.mean(axis=1)
correlation_i.loc['mean'] = correlation_i.mean(axis=0)
correlation_c['mean'] = correlation_c.mean(axis=1)
correlation_c.loc['mean'] = correlation_c.mean(axis=0)

correlation_i['weight'] = correlation_i['mean'] / correlation_i['mean'].iloc[:-1].sum()
correlation_c['weight'] = correlation_c['mean'] / correlation_c['mean'].iloc[:-1].sum()

weight = pd.concat([correlation_i['weight'], correlation_c['weight']], axis=1)
weight['average'] = weight.mean(axis=1)
weight.columns = ['interestingness', 'creativity', 'average']

weight.to_csv('./results/weights.csv')