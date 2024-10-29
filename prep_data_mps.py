import torch
import s_rd
from tokenizer_class import Tokenizer 

device = torch.device("mps")
# confirming correctly reading data
file_path_train='japanese_train.jsonl'
file_path_val='japanese_val.jsonl'

# reading data for AI generation of summary (for learning)
texts,summaries=s_rd.read_data(file_path_train)
train_data=""
for text, summary in zip(texts,summaries):
  train_data=train_data+"<BOS>"+text+"<SUMMARY>"+summary+"<EOS>"
vocab=Tokenizer.create_vocab(train_data)
tokenizer=Tokenizer(vocab)
vocab_size=len(vocab)

# reading data for AI generation of summary (for estimation/developing)
texts,summaries=s_rd.read_data(file_path_val)
val_data=""
for text, symmary in zip(texts,summaries):
  val_data=val_data+"<BOS>"+text+"<SUMMARY>"+summary+"<EOS>"

train_data=torch.tensor(tokenizer.encode(train_data),device=device,dtype=torch.long)
val_data=torch.tensor(tokenizer.encode(val_data),device=device,dtype=torch.long)

print(train_data)
print(val_data)