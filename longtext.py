import torch
import torch.nn as nn
import argparse
import numpy as np
from tqdm import tqdm
import os
from models import text as text_module

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainer', action='store_true', help='Whether to train the model')
    parser.add_argument('--sampler', action='store_true', help='Whether to sample from the trained model')
    parser.add_argument('--tester', action='store_true', help='Whether to test the trained model on saved generated text')

    parser.add_argument('--path', type=str, default='...', help='Path to load the trained model for sampling')
    parser.add_argument('--max_tokens', type=int, default=800, help='Maximum number of tokens to generate during sampling')
    parser.add_argument('--bs', type=int, default=64, help='Batch size for training')
    parser.add_argument('--block_size', type=int, default=32, help='Maximum context length for predictions, means length = block_size + 1')
    parser.add_argument('--max_iters', type=int, default=5000, help='Maximum number of iterations for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--device', type=str, default='3', help='Device to use for training, e.g., "-1" for CPU or positive integer for GPU')
    parser.add_argument('--checkpoint', type=int, default=500, help='Interval for saving model checkpoints during training')
    parser.add_argument('--dtype', type=str, default='float32', help='Data type for model parameters and computations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--show', type=int, default=1, help='the interval to show training loss')
    parser.add_argument('--SAVE_PATH', type=str, default='./LongtextResults/', help='Path to save training results and checkpoints')
    parser.add_argument('--text', type=str, default='', help='the decription of the this test')
    ## model hyperparameters
    parser.add_argument('--model', type=str, default='qcam', help='Model type to use')

    # QCAM/EQCAM specific arguments
    parser.add_argument('--depth', type=int, default=10, help='Number of circuit layers')
    parser.add_argument('--proj', type=str, default='cx', help='Projection gate type')
    parser.add_argument('--gate', type=str, default='rxry', help='Single qubit gate type')
    parser.add_argument('--reset', action='store_true', help='Whether to reset qubits between layers')

    # EMADE
    parser.add_argument('--activation', type=str, default='softmax', help='Activation function for EMADE model')
    
    args = parser.parse_args()
    mydevice = 'cpu' if args.device == '-1' else f'cuda:{args.device}'
    mydtype = torch.float32 if args.dtype == 'float32' else torch.float64
    if args.model == 'qcam' or args.model == 'eqcam':
        args.SAVE_PATH = args.SAVE_PATH + f'{args.model}/bs{args.bs}_block{args.block_size}_lr{args.lr}_depth{args.depth}_seed{args.seed}{args.text}'
    elif args.model == 'made' or args.model == 'emade':
        args.SAVE_PATH = args.SAVE_PATH + f'{args.model}/bs{args.bs}_block{args.block_size}_lr{args.lr}_{args.activation}_seed{args.seed}{args.text}'
    if not os.path.exists(args.SAVE_PATH):
        os.makedirs(args.SAVE_PATH)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    return args, mydevice, mydtype

def save_info(texter, args):
    with open(f'{args.SAVE_PATH}/info.toml', 'w') as f:
        f.write(f"[model_info]\n")
        f.write(f"name = \"tinyshakespeare\"\n")
        f.write(f"block_size = {args.block_size}\n")
        f.write(f"batch_size = {args.bs}\n")
        f.write(f"checkpoint = {args.checkpoint}\n")
        f.write(f"num_qubits = {texter.qubits}\n")
        f.write(f"lr = {args.lr}\n")
        f.write(f"max_iters = {args.max_iters}\n")
        f.write(f"seed = {args.seed}\n")
        f.write(f'model = "{args.model}"\n')
        if args.model == 'qcam' or args.model == 'eqcam':
            f.write(f"depth = {args.depth}\n")
            f.write(f'reset = {"true" if args.reset else "false"}\n')
            f.write(f'proj = "{args.proj}"\n')
            f.write(f'gate = "{args.gate}"\n')
        elif args.model == 'emade' or args.model == 'made':
            f.write(f'activation = "{args.activation}"\n')
        
    print(f'Training: batch size={args.bs}, block size={args.block_size}, max iters={args.max_iters}')
    if args.checkpoint > 0:
        if not os.path.exists(args.SAVE_PATH+'/checkpoints'):
            os.makedirs(args.SAVE_PATH+'/checkpoints')

def trainner(args, device, dtype):
    # /data/miyan/qcam_gen/data/dialogues/train.txt
    # source_text = text_module.TextLoader('./data/tinyshakespeare.txt', qubits=6, device=device)
    source_text = text_module.TextLoader('./data/dialogues/train.txt', qubits=5, device=device)
    source_text_test = text_module.TextLoader('./data/dialogues/test.txt', qubits=5, device=device)
    save_info(source_text, args) # save infos
    model = None
    optimizer = None
    train_data = None
    val_data = None
    get_batch = None
    if args.model == 'qcam':
        from qcam_seq import circuit
        model = circuit.Circuit(
            length=args.block_size+1,
            num_qubits=source_text.qubits, 
            mydevice=device, mydtype=dtype, 
            depth=args.depth, reset=args.reset, proj=args.proj, gate=args.gate
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        # train_data, val_data = text_module.preprocess(source_text, n=-1, type='bit')
        train_data, val_data = text_module.preprocess(source_text, source_text_test, type='bit')
        get_batch = lambda data, block_size, batch_size: text_module.get_batch(data, block_size, batch_size, type='x')[0]
    elif args.model == 'eqcam':
        from qcam_seq import ecircuit
        model = ecircuit.Circuit(
            embedding_dim=2**source_text.qubits,
            length=args.block_size+1,
            num_qubits=source_text.qubits, 
            mydevice=device, mydtype=dtype, 
            depth=args.depth, reset=args.reset, proj=args.proj, gate=args.gate
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        train_data, val_data = text_module.preprocess(source_text, source_text_test, type='bit')
        get_batch = lambda data, block_size, batch_size: text_module.get_batch(data, block_size, batch_size, type='x')[0]
    elif args.model == 'made':
        from models.made import MADE
        model = MADE(
            features=(args.block_size+1)*source_text.qubits,
            hidden_layers=0,
            activation="sigmoid",
            device=device,
            dtype=dtype
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        def get_batch(data, block_size, batch_size, device):
            # generate a small batch of data of inputs x 
            length = block_size + 1
            ix = torch.randint(len(data) - length, (batch_size,))
            x = torch.stack([data[i:i+length] for i in ix])
            x= x.to(device)
            return x.int().reshape(batch_size, -1)
    
    elif args.model == 'emade':
        from models.emade import EmbeddingMADE
        model = EmbeddingMADE(
            length=args.block_size+1,
            num_embeddings=2**source_text.qubits,
            embedding_dim=2**source_text.qubits,
            activation="softmax" # or "cosinesquare",
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # train_data, val_data = text_module.preprocess(source_text, n=-1, type='token')
        train_data, val_data = text_module.preprocess(source_text, source_text_test, type='token')
    
        get_batch = lambda data, block_size, batch_size: text_module.get_batch(data, block_size, batch_size, type='x')[0]
   
    elif args.model == 'transformer':
        ...
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    assert model is not None and optimizer is not None, "Model and optimizer must be initialized"
    
    train_record = []
    val_record = []
    for iter in (pbar := tqdm(range(args.max_iters), desc=f"Training lr:{args.lr}")):
        # every eval_interval iterations evaluate the loss on train and val sets
        with torch.no_grad():
            x = get_batch(val_data, args.block_size, args.bs)
            test_log_likelihoods = model(x, condition=True)
            
            test_loss = -torch.mean(test_log_likelihoods) / args.block_size
        optimizer.zero_grad()
        x = get_batch(train_data, args.block_size, args.bs)
        log_likelihoods = model(x, condition=True)
        train_loss = -torch.mean(log_likelihoods) / args.block_size
        train_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_record.append(train_loss.item())
        val_record.append(test_loss.item())
        if iter % args.show == 0:
            pbar.set_description(f"Train NLL: {train_record[-1]:.6f}, Val NLL: {val_record[-1]:.6f}")
        if args.checkpoint > 0 and iter % args.checkpoint == 0 and iter > 0:
            torch.save(model.state_dict(), f'{args.SAVE_PATH}/checkpoints/iter_{iter}.pt')
    torch.save(model.state_dict(), f'{args.SAVE_PATH}/model_final.pth')
    np.savetxt(f'{args.SAVE_PATH}/train_record.txt', np.array(train_record))
    np.savetxt(f'{args.SAVE_PATH}/val_record.txt', np.array(val_record))
    return train_record, val_record, model

### ---- Sampling Process ---- ###
def generate(model, max_new_tokens, block_size, type='token'):
    # first length data:
    # this is just for embedding
    device = next(model.parameters()).device
    # test 
    
    if type == 'token':
        generated_data = model.cond_sample(prompt=torch.zeros((1,), dtype=torch.int32).to(device), num_samples=1)
        # generated_data = model.cond_sample(prompt=torch.tensor([62, 56, 42, 1, 50, 42, 1, 52, 58], dtype=torch.int32).to(device), num_samples=1)
    else:
        generated_data = model.cond_sample(prompt=torch.zeros((1, model.num_qubits), dtype=torch.int32).to(device), num_samples=1)
        # generated_data = model.cond_sample(
        #     prompt = torch.tensor([[0, 0, 0, 0, 0, 0], # = 0
        #                          [0,0,0,0,0,0], # = 0
        #                            [0, 1, 1, 1, 1, 1], # = 31
        #                            [1, 0, 1, 1, 0, 1], # = 45
        #                            [1, 0, 1, 0, 1, 0], # = 42
        #                         #    [1, 1, 1, 0, 0, 0] # = 56
        #                            ], dtype=torch.int32).to(device),
        # )
    # print(generated_data)
    # exit()
    range_text = block_size
    while generated_data.shape[0] < max_new_tokens:
        condition_samples = model.cond_sample(prompt=generated_data[-range_text:], num_samples=1)
        new = condition_samples[range_text:range_text+1] # the last new token
        # print(new.shape)
        generated_data = torch.cat((generated_data, new), dim=0)
    print(generated_data.shape)
    return generated_data

# def bin_to_int(x_bin):
#     assert x_bin.shape[-1] == 6, "Input x_bin must have last dimension of size 6"
#     device = x_bin.device
#     powers = torch.tensor([32, 16, 8, 4, 2, 1], device=device, dtype=x_bin.dtype)
#     weighted_bits = x_bin * powers
#     x_int = weighted_bits.sum(dim=-1)
#     return x_int.long()

def bin_to_int(x_bin):
    assert x_bin.shape[-1] == 5, "Input x_bin must have last dimension of size 6"
    device = x_bin.device
    powers = torch.tensor([16, 8, 4, 2, 1], device=device, dtype=x_bin.dtype)
    weighted_bits = x_bin * powers
    x_int = weighted_bits.sum(dim=-1)
    return x_int.long()

def sampler(args, device, dtype):
    model = None
    import tomllib
    data_type = None
    with open(f'{args.path}/info.toml', 'rb') as f:
        info = tomllib.load(f)
    # source_text = text_module.TextLoader('./data/tinyshakespeare.txt', qubits=6, device=device)
    source_text = text_module.TextLoader('./data/dialogues/train.txt', qubits=5, device=device)
    if info['model_info']['model'] == 'eqcam':
        from qcam_seq import ecircuit
        model = ecircuit.Circuit(
            embedding_dim=2**info['model_info']['num_qubits'],
            length=info['model_info']['block_size']+1,
            num_qubits=info['model_info']['num_qubits'], 
            mydevice=device, mydtype=dtype, 
            depth=info['model_info']['depth'], reset=info['model_info']['reset'], proj=info['model_info']['proj'], gate=info['model_info']['gate']
        ).to(device)
        model.load_state_dict(torch.load(os.path.join(args.path, 'model_final.pth'), map_location=device))
        data_type = 'bit'
    elif info['model_info']['model'] == 'emade':
        from models.emade import EmbeddingMADE
        model = EmbeddingMADE(
            length=info['model_info']['block_size']+1,
            num_embeddings=2**info['model_info']['num_qubits'],
            embedding_dim=2**info['model_info']['num_qubits'],
            activation=info['model_info']['activation']
        ).to(device)
        model.load_state_dict(torch.load(os.path.join(args.path, 'model_final.pth'), map_location=device))
        data_type = 'token'
    else:
        raise ValueError(f"Sampling for model {info['model_info']['model']} is not implemented yet.")
    model.eval()
    generated_data = generate(model, args.max_tokens, info['model_info']['block_size'], type=data_type)
    if data_type == 'bit':
        generated_data = bin_to_int(generated_data.squeeze()).cpu().numpy().tolist()
    else:
        generated_data = generated_data.squeeze().cpu().numpy().tolist()
    generated_text = source_text.decode_fromint(generated_data)
    print(generated_text)
    # save to file
    with open(f'{args.path}/generated_text_seed{args.seed}.txt', 'w', encoding='utf-8') as f:
        f.write(generated_text)
    np.savetxt(f'{args.path}/generated_data_seed{args.seed}.txt', np.array(generated_data), fmt='%d')
    
def tester(args, device, dtype):
    model = None
    import tomllib
    data_type = None
    with open(f'{args.path}/info.toml', 'rb') as f:
        info = tomllib.load(f)
    # source_text = text_module.TextLoader('./data/tinyshakespeare.txt', qubits=6, device=device)
    source_text = text_module.TextLoader('./data/dialogues/train.txt', qubits=5, device=device)
    if info['model_info']['model'] == 'eqcam':
        from qcam_seq import ecircuit
        model = ecircuit.Circuit(
            embedding_dim=2**info['model_info']['num_qubits'],
            length=info['model_info']['block_size']+1,
            num_qubits=info['model_info']['num_qubits'], 
            mydevice=device, mydtype=dtype, 
            depth=info['model_info']['depth'], reset=info['model_info']['reset'], proj=info['model_info']['proj'], gate=info['model_info']['gate']
        ).to(device)
        model.load_state_dict(torch.load(os.path.join(args.path, 'model_final.pth'), map_location=device))
        data_type = 'bit'
    elif info['model_info']['model'] == 'emade':
        from models.emade import EmbeddingMADE
        model = EmbeddingMADE(
            length=info['model_info']['block_size']+1,
            num_embeddings=2**info['model_info']['num_qubits'],
            embedding_dim=2**info['model_info']['num_qubits'],
            activation=info['model_info']['activation']
        ).to(device)
        model.load_state_dict(torch.load(os.path.join(args.path, 'model_final.pth'), map_location=device))
        data_type = 'token'
    else:
        raise ValueError(f"Sampling for model {info['model_info']['model']} is not implemented yet.")

    with open(f'{args.path}/generated_text_seed42.txt', 'r', encoding='utf-8') as f:
        test_text = f.read()
    test_data = source_text.encode_toint(test_text)
    if data_type == 'bit':
        test_data = (torch.tensor(test_data, dtype=torch.long, device=device).unsqueeze(-1) >> source_text.shift) & 1
    else:
        test_data = torch.tensor(test_data, dtype=torch.long, device=device)
    model.eval()
    length = info['model_info']['block_size'] + 1
    ix = torch.arange(0, len(test_data) - length)
    x = torch.stack([test_data[i:i+length] for i in ix])
    with torch.no_grad():
        log_likelihoods = model(x, condition=True)
        test_loss = -torch.mean(log_likelihoods) / info['model_info']['block_size']
    print(f'Test NLL: {test_loss.item():.6f}')

if __name__ == "__main__":
    args, device, dtype = set_args()
    if args.trainer:
        train_record, val_record, model = trainner(args, device, dtype)
    if args.sampler:
        sampler(args, device, dtype)
    if args.tester:
        tester(args, device, dtype)