import torch
import s_rd
from tokenizer_class import Tokenizer 

# data for generating Token+ID_pair
file_path_train = 'japanese_train.jsonl'
# reading data set and combining them to one string
texts,summaries=s_rd.read_data(file_path_train)
dataset="".join(texts+summaries)
# 1. making vocabulary
vocab=Tokenizer.create_vocab(dataset)
# 2. initialization of tokenizer
tokenizer=Tokenizer(vocab)
# 3. encoding first text and summary for example
sample_text=texts[0]
#print(sample_text)
sample_summary=summaries[0]
#print(sample_summary)

# encoding
encoded_text=tokenizer.encode(sample_text)
#print("encoded text: ",encoded_text)
encoded_summary=tokenizer.encode(sample_summary)
#print("encoded summary: ",encoded_summary)

#decoding
decoded_text=tokenizer.decode(encoded_text)
#print("encoded text: ",decoded_text)
decoded_summary=tokenizer.decode(encoded_summary)
#print("encoded summary: ",decoded_summary)

