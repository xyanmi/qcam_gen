import torch

'''
This file contains some preprocess functions for training to generate a long text.
'''

class TextLoader:
    def __init__(self, data_path, qubits=6, device='cpu'):
        self.data_path = data_path
        self.qubits = qubits
        self.mydevice = device
        with open(data_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        chars = sorted(list(set(self.text))) # the chars type in text
        self.real_vocab_size = len(chars)
        self.chars = chars + ['[UNK]'] * (2**self.qubits-self.real_vocab_size) # fill the space
        self.vocab_size = len(self.chars)
        print(f'Loading text from {self.data_path}\n-- orginal vocab size: {self.real_vocab_size}, adjusted vocab size: {self.vocab_size}')
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.shift = torch.tensor([self.qubits-i-1 for i in range(0, self.qubits)], device=device)
        

    def encode_toint(self, s):
        return [self.stoi[c] for c in s] 
    def decode_fromint(self, l):
        return ''.join([self.itos[i] for i in l])
    

# def preprocess(texter, n=-1, type='bit'):
#     '''
#     Preprocess the text data, type:
#         - 'bit', sequence with (text_length, qubits)
#         - 'char', sequence with (text_length, )
#     n is the training length
#     '''
#     data = torch.tensor(texter.encode_toint(texter.text), dtype=torch.long, device=texter.mydevice)
#     if type == 'bit':
#         data = (data.unsqueeze(-1) >> texter.shift) & 1
#     n = int(0.9*len(data)) if n < 0 else n
#     train_data = data[:n]
#     val_data = data[n:]
#     return train_data, val_data # all on device

def preprocess(texter_train, texter_test, type='bit'):
    '''
    Preprocess the text data, type:
        - 'bit', sequence with (text_length, qubits)
        - 'char', sequence with (text_length, )
    n is the training length
    '''
    data_train = torch.tensor(texter_train.encode_toint(texter_train.text), dtype=torch.long, device=texter_train.mydevice)
    data_test = torch.tensor(texter_test.encode_toint(texter_test.text), dtype=torch.long, device=texter_test.mydevice)
    if type == 'bit':
        data_train = (data_train.unsqueeze(-1) >> texter_train.shift) & 1
        data_test = (data_test.unsqueeze(-1) >> texter_test.shift) & 1
    return data_train, data_test # all on device
    n = int(0.9*len(data)) if n < 0 else n
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data # all on device

def get_batch(data, block_size, batch_size, type='x'):
    # generate a small batch of data of inputs x or x, y
    if type == 'x':
        length = block_size + 1
        ix = torch.randint(len(data) - length, (batch_size,))
        x = torch.stack([data[i:i+length] for i in ix])
        return x, None
    elif type == 'xy':
        ix = torch.randint(0, len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y
    else:
        raise ValueError("type must be 'x' or 'xy'")
    
