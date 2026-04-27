"""
Quantum Circuit Model Trainer and Sampler

This script allows for training a new quantum circuit model or generating samples
from a pre-trained model based on command-line arguments.
"""

import os
import sys
import argparse
import tomllib
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

# To handle local module imports, it's common to add the project root to the path.
# For a more robust solution, consider making 'qcam' an installable package.
try:
    from qcam.circuit import Circuit
except ImportError:
    # Add project root to path to allow for local 'qcam' import
    project_root = os.path.dirname(os.path.abspath(''))
    sys.path.insert(0, project_root)
    from qcam.circuit import Circuit


def parse_arguments():
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(description="Quantum Circuit Model Trainer and Sampler")

    # --- General Settings ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index to use (e.g., '0', '1'). Uses CPU if CUDA is not available.")
    parser.add_argument("--path", type=str, default="", help="Path to a saved model directory for sampling.")
    parser.add_argument("--bs", type=int, default=0, help="Batch size for training. If 0, the entire dataset is used as one batch.")
    
    # --- Mode Selection ---
    parser.add_argument("--trainer", action="store_true", help="Enable this flag to train a new model.")
    parser.add_argument("--sampler", action="store_true", help="Enable this flag to sample from a trained model specified by --path.")

    # --- Training Hyperparameters ---
    parser.add_argument("--data", type=str, default="", help="Name of the dataset configuration in 'config.toml'.")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for the Adam optimizer.")
    parser.add_argument("--test", type=int, default=0, help="wether use test set to validate the model, 0 is no, positive integer is the rate between train set and test set.")
    parser.add_argument("--checkpoint", type=int, default=0, help="Checkpoint saving interval (0 disables checkpointing).")
    parser.add_argument("--test_bs", type=int, default=0, help="Batch size for test set, if 0, use full test set.")

    # --- Model Architecture ---
    parser.add_argument('--n', type=int, default=3, help="Number of qubits in the circuit.")
    parser.add_argument("--depth", type=int, default=1, help="Depth of each circuit block.")
    parser.add_argument("--reset", action="store_true", help="If set, reset data qubits to the |0> state after each block.")
    parser.add_argument("--proj", type=str, default="cx", choices=["cnot", "cz", "swap"], help="Projection gate to use.")
    parser.add_argument("--gate", type=str, default="rx", choices=["rx", "ry", "rxry"], help="Single qubit gate to use.")
    
    # --- Display Settings ---
    parser.add_argument("--show", type=int, default=1, help="Interval for displaying loss during training (in epochs).")
    parser.add_argument("--interval", type=int, default=1, help="Minimum time interval (in seconds) for updating the tqdm progress bar.")

    args = parser.parse_args()
    
    # --- Argument Validation ---
    if not args.trainer and not args.sampler:
        parser.error("At least one mode must be selected: --trainer or --sampler.")
    if args.sampler and not args.path:
        parser.error("--path must be provided when --sampler is enabled.")
    if args.trainer and not args.data:
        parser.error("--data must be provided when --trainer is enabled.")

    return args


def setup_environment(seed, device_id):
    """Sets random seeds and determines the computation device."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        device = torch.device(f'cuda:{device_id}')
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def prepare_training_artifacts(args: argparse.Namespace, config) -> str:
    """Creates directories and saves configuration files for a training run."""
    base_path = os.path.join("results", args.data)
    run_name = (
        f'n_{args.n}_depth{args.depth}_reset({args.reset})_proj({args.proj})_'
        f'gate({args.gate})_lr{args.lr}_epochs{args.epochs}'
    )
    save_path = os.path.join(base_path, run_name)
    os.makedirs(save_path, exist_ok=True)

    # Save training info to a TOML file
    info_path = os.path.join(save_path, 'info.toml')
    with open(info_path, 'w') as f:
        f.write("[model]\n")
        f.write(f"name = \"{args.data}\"\n")
        f.write(f"features = {config['features']}\n")
        f.write(f"num_qubits = {args.n}\n")
        f.write(f"depth = {args.depth}\n")
        f.write(f'reset = {"true" if args.reset else "false"}\n')
        f.write(f'proj = "{args.proj}"\n')
        f.write(f'gate = "{args.gate}"\n')
        f.write(f"lr = {args.lr}\n")
        f.write(f"epochs = {args.epochs}\n")
        f.write(f"seed = {args.seed}\n")
        f.write(f"batch_size = {args.bs}\n")
    return save_path


def train_model(args, device):
    """Handles the model training process."""
    print("\n--- Starting Model Training ---")

    # --- Load Data ---
    try:
        with open("./config.toml", "rb") as f:
            config = tomllib.load(f)[args.data]
    except (FileNotFoundError, KeyError):
        print(f"Error: Could not find dataset '{args.data}' in config.toml.")
        return
    save_path = prepare_training_artifacts(args, config)
    features = config['features']
    if config['path'].endswith('.npy'):
        data = np.load(config['path']).reshape(-1, config['features'])
    else:
        data = np.loadtxt(config['path'], delimiter=',')
        # data = data.reshape(-1, config['L'], config['L'])
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    print(f"Loaded Dataset: {config['name']}, Features: {features}, Shape: {data_tensor.shape}")


    model = Circuit(
        features=features, 
        num_qubits=args.n, 
        mydevice=device, 
        mydtype=torch.float32,
        depth=args.depth,
        reset=args.reset,
        proj=args.proj,
        gate=args.gate
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # divide data into train and test set if args.test > 0

    if args.test > 0:
        index = int( len(data) * args.test / (args.test + 1) )
        train_data, test_data = data[:index], data[index:]
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.bs if args.bs > 0 else len(train_data), shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs if args.test_bs > 0 else len(test_data), shuffle=False)
        print(f"Train set: {train_data.shape}, Test set: {test_data.shape}")
    else:
        train_data = data
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.bs if args.bs > 0 else len(train_data), shuffle=True)
        test_dataloader = None
        print(f"Train set: {train_data.shape}")
    
    # --- Training Loop ---
    nll_history = []
    nll_testing = []
    for epoch in (pbar := tqdm(range(args.epochs), desc=f"Training (LR:{args.lr})", mininterval=args.interval)):
        epoch_loss = 0.0
        for batch in test_dataloader if test_dataloader else []:
            batch = batch.to(dtype=torch.float32, device=device)
            with torch.no_grad():
                log_likelihoods = model(batch)
                test_loss = -torch.mean(log_likelihoods)
                epoch_loss += test_loss.item() * batch.shape[0]
        if test_dataloader: nll_testing.append(epoch_loss / len(test_data))
        epoch_loss = 0.0
        for batch in train_dataloader:
            batch = batch.to(dtype=torch.float32, device=device)
            optimizer.zero_grad()
            log_likelihoods = model(batch)
            loss = -torch.mean(log_likelihoods)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * batch.shape[0]
        nll_history.append(epoch_loss / len(train_data))
        if epoch % args.show == 0:
            if test_dataloader:
                pbar.set_description(f"Train NLL: {nll_history[-1]:.6f}, Test NLL: {nll_testing[-1]:.6f}")
            # pbar.set_postfix(Training_nll=f"{epoch_loss:.6f}")
            else:
                pbar.set_description(f"Train NLL: {nll_history[-1]:.6f}")
        
        # --- Checkpoint Saving ---
        if args.checkpoint > 0 and (epoch + 1) % args.checkpoint == 0:
            chk_path = os.path.join(save_path, 'checkpoints')
            os.makedirs(chk_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(chk_path, f'model_e{epoch+1}.pth'))

    
    print(f"Final NLL: {nll_history[-1]:.6f}")

    # Save model state
    torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
    print(f"Model saved to {os.path.join(save_path, 'model.pth')}")

    # Save NLL history
    np.savetxt(os.path.join(save_path, 'nll.training'), np.array(nll_history), fmt='%.6f')
    print(f"Training info and NLL history saved in {save_path}")
    np.savetxt(os.path.join(save_path, 'nll.testing'), np.array(nll_testing), fmt='%.6f')
    # return model


def sample_from_model(args, device):
    """Handles loading a model and generating samples."""
    print("\n--- Starting Model Sampling ---")
    model_path = args.path

    if not os.path.isdir(model_path) or not os.path.exists(os.path.join(model_path, 'info.toml')):
        print(f"Error: Model directory or 'info.toml' not found at '{model_path}'")
        return

    # --- Load Model Configuration ---
    with open(os.path.join(model_path, 'info.toml'), "rb") as f:
        config = tomllib.load(f)['model']
    
    print("Loaded model configuration:")
    for key, val in config.items():
        print(f"  - {key}: {val}")
        
    # --- Recreate Model and Load State ---
    model = Circuit(
        features=config['features'], 
        num_qubits=config['num_qubits'], 
        mydevice=device, 
        mydtype=torch.float32,
        depth=config['depth'],
        reset=config['reset'],
        proj=config['proj'],
        gate=config['gate']
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth'), map_location=device))
    model.eval()
    print("Model state loaded successfully.")

    # --- Generate and Save Samples ---
    num_samples = args.bs if args.bs > 0 else 100
    new_samples = model.sample(num_samples=num_samples)
    print(f"\nGenerated {num_samples} samples (shape: {new_samples.shape})")

    save_file = f'generated_samples_ns{num_samples}_seed{args.seed}.npy'
    save_filepath = os.path.join(model_path, save_file)
    np.save(save_filepath, new_samples.cpu().numpy())
    print(f"Generated samples saved to {save_filepath}")


def main():
    """Main function to orchestrate training and sampling."""
    args = parse_arguments()
    device = setup_environment(args.seed, args.device)

    if args.trainer:
        train_model(args, device)

    if args.sampler:
        sample_from_model(args, device)


if __name__ == "__main__":
    main()


# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--n', type=int, default=3)
# parser.add_argument("--seed", type=int, default=42, help="Random seed")
# parser.add_argument("--device", type=str, default="0", help="Device to use (cuda:0, cpu, etc.)")
# parser.add_argument("--path", type=str, default="", help="Path to saved model")

# parser.add_argument("--trainer", action="store_true", help="Whether Train a new model")
# parser.add_argument("--epochs", type=int, default=2000, help="Number of epochs for training")
# parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for training")
# parser.add_argument("--data", type=str, default="", help="toml variable")
# parser.add_argument("--bs", type=int, default=0, help="batch size for training")
# parser.add_argument("--depth", type=int, default=1, help="circuit block depth")
# parser.add_argument("--wd", action="store_true", help="Whether add data encoding in circuit, add this term mean dont add data(without data)")
# parser.add_argument("--show", type=int, default=1, help="loss display interval")
# parser.add_argument("--interval", type=int, default=1, help="tqdm display interval(s)")
# parser.add_argument("--reset", action="store_true", help="Whether reset the data qubits")
# parser.add_argument("--proj", type=str, default="cx", help="Projection gate to use: cx, cz, swap")
# parser.add_argument("--gate", type=str, default="rx", help="Single qubit gate to use: rx, ry, rxry")

# parser.add_argument("--sampler", action="store_true", help="Whether sample from trained model")

# args = parser.parse_args()

# assert args.sampler or args.trainer, "Please select at least one mode: --sampler or --trainer"
# import os
# import sys
# dirs = os.path.dirname(os.path.abspath(''))
# sys.path.insert(0, dirs)

# import torch
# import numpy as np
# torch.manual_seed(args.seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(args.seed)

# # Set device
# device = torch.device('cuda:' + args.device if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# def Trainer(data_toml, epochs, num_qubits, lr, depth, add_data, reset, proj, gate):
#     assert data_toml != "", "Please provide the toml variable for training data"

#     from qcam.circuit import Circuit
#     import torch.optim as optim
#     import tomllib
#     from tqdm import tqdm
    
#     with open("./config.toml", "rb") as f:
#         config = tomllib.load(f)[data_toml]

#     features = config['features']
#     data = np.loadtxt(config['path'], delimiter=',')
#     data = torch.tensor(data, dtype=torch.int32).to(device)
#     print('Loading Info: ', config['name'], 'Features:', features, 'Data shape:', data.shape)
#     model = Circuit(
#         features=features, 
#         num_qubits=num_qubits, 
#         mydevice=device, 
#         mydtype=torch.float32,
#         depth=depth,
#         add_data=add_data,
#         reset=reset,
#         proj=proj,
#         gate=gate
#     ).to(device)
#     nll = []
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     # optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=20,)
#     # optimizer = optim.SGD(model.parameters(), lr=lr)
#     dataloader = torch.utils.data.DataLoader(data, batch_size=args.bs if args.bs > 0 else len(data), shuffle=True)
#     for e in (pbar := tqdm(range(epochs), desc=f"Training lr:{lr}", mininterval=args.interval)):
#         epoch_loss = 0.0
#         for batch in dataloader:
#             optimizer.zero_grad()

#             log_likelihoods = model(batch.to(device))
#             loss = -torch.mean(log_likelihoods)
#             optimizer.zero_grad()
#             loss.backward()
#             # print(max([p.grad.abs().max().item() for p in model.parameters() if p.grad is not None]))
#             optimizer.step()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             epoch_loss += loss.item() * batch.shape[0]
#             # print(loss.item())
#         epoch_loss /= len(data)
#         if e % args.show == 0:
            
#             pbar.set_postfix(NLL=f"{epoch_loss:.6f}")
#         nll.append(epoch_loss)
#     save_path = "./results" + f"/{data_toml}"
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     save_path = save_path + f'/n{num_qubits}_depth{depth}_wd({not add_data})_reset({reset})_proj({proj})_gate({gate})_lr{lr}_epochs{epochs}'
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     # save_path = f"./results/model_parameters_n{num_qubits}_depth{depth}_ad{add_data}_lr{lr}_epochs{epochs}.pth"
#     torch.save(model.state_dict(), save_path + '/model.pth')
#     print(f"Model parameters saved to {save_path}/model.pth")
#     # save info
#     with open(save_path + '/info.toml', 'w') as f:
#         f.write(f"[model]\n")
#         f.write(f"name = \"{data_toml}\"\n")
#         f.write(f"features = {features}\n")
#         f.write(f"num_qubits = {num_qubits}\n")
#         f.write(f"depth = {depth}\n")
#         f.write(f'reset = {int(reset)}\n')
#         f.write(f'proj = "{proj}"\n')
#         f.write(f'gate = "{gate}"\n')
#         f.write(f"without_data = {int(not add_data)}\n")
#         f.write(f"lr = {lr}\n")
#         f.write(f"epochs = {epochs}\n")
#         f.write(f"seed = {args.seed}\n")
#     # save nll history
#     np.savetxt(save_path+'/nll.training', np.array(nll), fmt='%.6f')
#     print(f"NLL history saved to {save_path}/nll.training")
#     return model

# def Sampler(model_path, num_samples):
#     from qcam.circuit import Circuit
#     import torch
#     import tomllib
    
#     with open(model_path + '/info.toml', "rb") as f:
#         config = tomllib.load(f)
#         features = config['model']['features']
#         num_qubits = config['model']['num_qubits']
#         depth = config['model']['depth']
#         add_data = not int(config['model']['without_data'])
#         reset = bool(int(config['model']['reset']))
#         proj = config['model']['proj']
#         gate = config['model']['gate']
#     print('Loading Info: ', config['model']['name'], 'Features:', features, 'num_qubits:', num_qubits, 'depth:', depth, 'with_data:', add_data, 'reset:', reset, 'proj:', proj, 'gate:', gate)
#     model = Circuit(
#         features=features, 
#         num_qubits=num_qubits, 
#         mydevice=device, 
#         mydtype=torch.float32,
#         depth=depth,
#         add_data=add_data,
#         reset=reset,
#         proj=proj,
#         gate=gate
#     ).to(device)
#     model.load_state_dict(torch.load(model_path + '/model.pth', map_location=device))
#     model.eval()
#     print("Model loaded successfully.")

#     new_samples = model.sample(num_samples=num_samples)
#     print("\nGenerated Samples (shape: {}):".format(new_samples.shape))
#     save_path = model_path + f'/generated_samples_ns{num_samples}_seed{args.seed}.data'
#     np.savetxt(save_path, new_samples.cpu().numpy(), fmt='%d', delimiter=',')
#     print(f"Generated samples saved to {save_path}")

# if args.trainer:
#     import os
#     if not os.path.exists('./results'):
#         os.makedirs('./results')
#     model = Trainer(
#         data_toml=args.data, 
#         epochs=args.epochs, 
#         num_qubits=args.n, 
#         lr=args.lr,
#         depth=args.depth,
#         add_data=not args.wd,
#         reset=args.reset,
#         proj=args.proj,
#         gate=args.gate
#     )
# if args.sampler:
#     assert args.path != "", "Please provide the path to the trained model using --path"
#     if not os.path.exists(args.path):
#         print(f"Error: Model file not found at '{args.path}'")
#         print("Please make sure you have a trained model saved at this location.")
#         exit()
#     Sampler(model_path=args.path, num_samples=args.bs if args.bs > 0 else 100)