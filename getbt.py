# extractionextraction of learning data: extraction sevral cases for batch processing
def get_batch(split):
  import torch
  from prep_data_sub import prep_d 
  blocksize=500  # treating until 500
  train_data,val_data=prep_d()
  # generate a small batch of data of inputs x and targets y
  data = train_data if split == 'train' else val_data
  # print(len(data))
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  # print(x[0])
  # print(y[0])
  x, y = x.to(device), y.to(device)
  return x, y
