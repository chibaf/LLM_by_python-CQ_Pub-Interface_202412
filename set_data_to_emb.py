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
embedding=nn.Embedding(num_embeddings=vocab_size,embedding_dim=emb_dim)

#set data into embedded layer
tokenizer=Tokenizer(vocab)
encoded_text=tokenizer.encode("犬と猫")
t_tokens=torch.tensor(data=encoded_text).unsqueeze(dim=0) # [9]->[1,9]
x=embedding(t_tokens) # [1,9]->[1,9,50] : embedding vectors
print(x) # given three characters are translated to embedding vectors respectively
print(x.shape) # matrix of size([1,3,300])