from os import listdir
import pandas as pd

paths = ['./datas/validation_set','./datas/weight_tuning_set']

for path in paths:
    datas = listdir(path)

    df = pd.DataFrame()
    for data in datas:
        temp = pd.read_json(f'{path}/{data}', lines=True)
        temp['transformation'] = data.split('.')[0]
        df = pd.concat([df, temp])
    
    save_name = path.split('/')[-1]

    df.to_json(f'./datas/total/{save_name}.json', orient='records', lines=True)