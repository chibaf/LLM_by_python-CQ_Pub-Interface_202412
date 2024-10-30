import torch
import s_rd
import torch.nn as nn
from tokenizer_class import Tokenizer 

device = 'mps' if torch.mps.is_available() else 'cpu'  # for apple silicon

# confirming correctly reading data
file_path_train='japanese_train.jsonl'
file_path_val='japanese_val.jsonl'

# reading data for AI generation of summary (for learning)
texts,summaries=s_rd.read_data(file_path_train)
train_data=""
for text, summary in zip(texts,summaries):
  train_data=train_data+"<BOS>"+text+"<SUMMARY>"+summary+"<EOS>"
vocab=Tokenizer.create_vocab(train_data)
# set parameters
vocab_size=len(vocab)
emb_dim=64 # size of embedding vector
# difinition of embedding layer
embddeing=nn.Embedding(num_embeddings=vocab_size,embedding_dim=emb_dim)
#tokenizer=Tokenizer(vocab)
#vocab_size=len(vocab)
#
## reading data for AI generation of summary (for estimation/developing)
#texts,summaries=s_rd.read_data(file_path_val)
#val_data=""
#for text, symmary in zip(texts,summaries):
#  val_data=val_data+"<BOS>"+text+"<SUMMARY>"+summary+"<EOS>"
#
#train_data=torch.tensor(tokenizer.encode(train_data),dtype=torch.long)
#val_data=torch.tensor(tokenizer.encode(val_data),dtype=torch.long)
#
#print(train_data)
#print(val_data)