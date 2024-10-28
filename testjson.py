import torch
import s_rd
device = 'mps' if torch.mps.is_available() else 'cpu'  # for apple silicon
file_path_train='japanese_train.jsonl'
file_path_val='japanese_val.jsonl'
texts,summaries=s_rd.read_data(file_path_train)
print('all text:')
print(texts[0])
print('summary:')
print(summaries[0])
