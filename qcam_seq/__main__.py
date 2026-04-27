import argparse
import os
import sys
import torch
import numpy as np
import tomllib
from tqdm import tqdm
import json

dirs = os.path.dirname(os.path.abspath(''))
sys.path.insert(0, dirs)
from qcam_seq.circuit import Circuit
from qcam_seq.__interact__ import run_interactive_console
import qcam_seq.__interact__ as qi


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Quantum Circuit Model Trainer and Sampler")

    # --- Group for General Settings ---
    general_group = parser.add_argument_group("General Settings")
    general_group.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    general_group.add_argument("--device", type=str, default="0", help="Device to use (e.g., '0' for cuda:0, 'cpu').")
    general_group.add_argument("--path", type=str, default="", help="Path to the saved model for sampling.")
    general_group.add_argument("--bs", type=int, default=0, help="Batch size. If 0, uses the full dataset for training or 100 for sampling.")

    # --- Group for Mode Selection ---
    mode_group = parser.add_argument_group("Mode Selection")
    mode_group.add_argument("--trainer", action="store_true", help="Activate training mode.")
    mode_group.add_argument("--sampler", action="store_true", help="Activate sampling mode.")
    mode_group.add_argument("--interact", action="store_true", help="Activate interactive console mode.")

    # --- Group for Training Settings ---
    train_group = parser.add_argument_group("Training Settings")
    train_group.add_argument("--epochs", type=int, default=2000, help="Number of training epochs.")
    train_group.add_argument("--lr", type=float, default=0.01, help="Learning rate for the Adam optimizer.")
    train_group.add_argument("--data", type=str, default="", help="TOML variable key for the dataset configuration.")
    train_group.add_argument("--depth", type=int, default=1, help="Depth of the circuit block.")
    train_group.add_argument("--show", type=int, default=1, help="Interval for displaying loss during training.")
    train_group.add_argument("--interval", type=int, default=1, help="Update interval for the tqdm progress bar (in seconds).")
    train_group.add_argument("--reset", action="store_true", help="Whether to reset data qubits after each step.")
    train_group.add_argument("--proj", type=str, default="cx", choices=["cnot", "cz", "swap"], help="Projection gate to use.")
    train_group.add_argument("--gate", type=str, default="rx", choices=["rx", "ry", "rxry"], help="Single qubit gate to use.")
    train_group.add_argument("--checkpoint", type=int, default=0, help="Checkpoint saving interval (0 disables checkpointing).")
    train_group.add_argument("--test", type=int, default=0, help="wether use test set to validate the model, 0 is no, positive integer is the rate between train set and test set.")
    train_group.add_argument("--test_bs", type=int, default=0, help="Batch size for test set, if 0, use full test set.")
    train_group.add_argument("--mask", action="store_true", help="whether use the mask in the loss function.")

    # --- Group for Sampling Settings ---
    sample_group = parser.add_argument_group("Sampling Settings")
    sample_group.add_argument("--prompt", type=str, default="", help="Prompt for text generation, must be json file name.")

    
    args = parser.parse_args()
    selected_modes = sum([args.trainer, args.sampler, args.interact])
    if selected_modes == 0:
        parser.error("No mode selected. Please choose one of --trainer, --sampler, or --interact.")
    if selected_modes > 1:
        parser.error("Multiple modes selected. Please choose only one of --trainer, --sampler, or --interact.")
    return args

def setup_environment(seed: int, device_id: str) -> torch.device:
    """Sets up random seeds and the computation device."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    device_str = f'cuda:{device_id}' if torch.cuda.is_available() and device_id != 'cpu' else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")
    return device

def prepare_training_artifacts(args: argparse.Namespace, config) -> str:
    """Creates directories and saves configuration files for a training run."""
    base_path = os.path.join("results_seq", args.data)
    run_name = (
        f'depth{args.depth}_reset({args.reset})_proj({args.proj})_'
        f'gate({args.gate})_lr{args.lr}_epochs{args.epochs}'
    )
    save_path = os.path.join(base_path, run_name)
    os.makedirs(save_path, exist_ok=True)

    # Copy vocabulary file
    vocab_src = os.path.join(config['path'], 'vocab.json')
    vocab_dst = os.path.join(save_path, 'vocab.json')
    if os.path.exists(vocab_src):
        os.system(f'cp "{vocab_src}" "{vocab_dst}"')

    # Save training info to a TOML file
    info_path = os.path.join(save_path, 'info.toml')
    with open(info_path, 'w') as f:
        f.write("[model]\n")
        f.write(f"name = \"{args.data}\"\n")
        f.write(f"length = {config['length']}\n")
        f.write(f"num_qubits = {config['qubits']}\n")
        f.write(f"depth = {args.depth}\n")
        f.write(f'reset = {"true" if args.reset else "false"}\n')
        f.write(f'proj = "{args.proj}"\n')
        f.write(f'gate = "{args.gate}"\n')
        f.write(f"lr = {args.lr}\n")
        f.write(f"epochs = {args.epochs}\n")
        f.write(f"seed = {args.seed}\n")
        
    return save_path

def run_trainer(args: argparse.Namespace, device: torch.device):
    """Handles the model training process."""
    assert args.data, "Please provide the TOML variable for training data using --data"

    with open("./config_seq.toml", "rb") as f:
        config = tomllib.load(f)[args.data]
    
    length, num_qubits = config['length'], config['qubits']
    data = np.load(os.path.join(config['path'], 'binary_sentences.npy'))
    data = torch.tensor(data, dtype=torch.int32).to(device)
    print(f"Loading Info: {config['name']}, Sequence length: {length}, num_qubits: {num_qubits}, data shape: {data.shape}")

    save_path = prepare_training_artifacts(args, config)

    model = Circuit(
        length=length, num_qubits=num_qubits, mydevice=device,
        mydtype=torch.float32, depth=args.depth, reset=args.reset,
        proj=args.proj, gate=args.gate
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
  
    nll_history = []
    nll_testing = []
    for epoch in (pbar := tqdm(range(args.epochs), desc=f"Training lr:{args.lr}", mininterval=args.interval)):
        epoch_loss = 0.0
        for batch in test_dataloader if test_dataloader else []:
            with torch.no_grad():
                mask = batch[:, :, -1].to(torch.float32) if args.mask else None
                data = batch[:, :, :-1] if args.mask else batch
                log_likelihoods = model(data, mask=mask)
                test_loss = -torch.mean(log_likelihoods)
                # nll_testing.append(test_loss.item())
                epoch_loss += test_loss.item() * batch.shape[0]
        if test_dataloader: nll_testing.append(epoch_loss / len(test_data))
        epoch_loss = 0.0
        for batch in train_dataloader:
            optimizer.zero_grad()
            mask = batch[:, :, -1].to(torch.float32) if args.mask else None
            data = batch[:, :, :-1] if args.mask else batch
            log_likelihoods = model(data, mask=mask)
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
        
        if args.checkpoint > 0 and (epoch + 1) % args.checkpoint == 0:
            chk_path = os.path.join(save_path, 'checkpoints')
            os.makedirs(chk_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(chk_path, f'model_e{epoch+1}.pth'))

    # Save final model and NLL history
    torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
    print(f"\nModel parameters saved to {os.path.join(save_path, 'model.pth')}")
    np.savetxt(os.path.join(save_path, 'nll.training'), np.array(nll_history), fmt='%.6f')
    print(f"NLL history saved to {os.path.join(save_path, 'nll.training')}")
    if test_dataloader:
        np.savetxt(os.path.join(save_path, 'nll.testing'), np.array(nll_testing), fmt='%.6f')
        print(f"Testing NLL history saved to {os.path.join(save_path, 'nll.testing')}")

def run_sampler(args: argparse.Namespace, device: torch.device):
    """Handles sampling from a trained model."""
    assert args.path, "Please provide the path to the trained model using --path"
    if not os.path.exists(args.path):
        sys.exit(f"Error: Model directory not found at '{args.path}'")

    with open(os.path.join(args.path, 'info.toml'), "rb") as f:
        config = tomllib.load(f)['model']

    print(f"Loading Info: {config['name']}, length: {config['length']}, num_qubits: {config['num_qubits']}, "
          f"depth: {config['depth']}, reset: {config['reset']}, proj: {config['proj']}, gate: {config['gate']}")
    
    model = Circuit(
        length=config['length'], num_qubits=config['num_qubits'], mydevice=device,
        mydtype=torch.float32, depth=config['depth'], reset=config['reset'],
        proj=config['proj'], gate=config['gate']
    ).to(device)

    model.load_state_dict(torch.load(os.path.join(args.path, 'model.pth'), map_location=device))
    model.eval()
    print("Model loaded successfully.")

    num_samples = args.bs if args.bs > 0 else 100
    new_samples = model.sample(num_samples=num_samples).cpu().numpy()

    print(f"\nGenerated Samples (shape: {new_samples.shape}):")
    save_file = f'generated_samples_ns{num_samples}_seed{args.seed}.npy'
    save_path = os.path.join(args.path, save_file)
    np.save(save_path, new_samples)
    print(f"Generated samples saved to {save_path}")

def run_conditonal_sampler(args: argparse.Namespace, device: torch.device):

    assert args.path, "Please provide the path to the trained model using --path"
    if not os.path.exists(args.path):
        sys.exit(f"Error: Model directory not found at '{args.path}'")
    assert args.prompt, "Please provide the prompt file name using --prompt"
    if args.prompt == "default": args.prompt = os.path.join(args.path, "prompt.json") # this is in the path
    assert args.prompt.endswith('.json'), "Prompt file must be a JSON file"
    with open(args.prompt, "r") as f:
        prompt_data = json.load(f)

    with open(os.path.join(args.path, 'info.toml'), "rb") as f:
        config = tomllib.load(f)['model']

    print(f"Loading Info: {config['name']}, length: {config['length']}, num_qubits: {config['num_qubits']}, "
          f"depth: {config['depth']}, reset: {config['reset']}, proj: {config['proj']}, gate: {config['gate']}")
    # ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyz .,?'")
    vocab = json.load(open(os.path.join(args.path, 'vocab.json'), 'r'))
    try:
        max_token = max(int(k) for k in vocab['token_to_char'].keys())
        qubit_length = max_token.bit_length()
        if qubit_length <= 5:
            is_token = False
        else:
            is_token = True
    except Exception as e:
        is_token = False
    print(f"Determined is_token: {is_token}, qubit_length: {qubit_length}")
    model = Circuit(
        length=config['length'], num_qubits=config['num_qubits'], mydevice=device,
        mydtype=torch.float32, depth=config['depth'], reset=config['reset'],
        proj=config['proj'], gate=config['gate']
    ).to(device)

    model.load_state_dict(torch.load(os.path.join(args.path, 'model.pth'), map_location=device))
    model.eval()
    print("Model loaded successfully.")
    samples = []
    for index, prompt in enumerate(prompt_data):
        prompt = prompt.lower()
        # if any(char not in ALLOWED_CHARS for char in prompt):
        #     raise ValueError(f"Error: Prompt contains invalid characters. Allowed characters are: {ALLOWED_CHARS}")
        try:
            input_data = qi.parse_single_sentence(vocab, prompt, is_token=is_token, qubit_length=qubit_length).to(device)
        except ValueError as e:
            print(e)
            continue
        with torch.no_grad():
            print(input_data)
            generated_samples = model.cond_sample(num_samples=1, prompt=input_data).cpu().numpy()
            generated_text = qi._single_samples_to_text(generated_samples, vocab, prompt_length = len(prompt))
            # print(generated_samples)
            samples.append([prompt, generated_text])

    save_file = f'prompt_complete_seed{args.seed}.json'
    save_path = os.path.join(args.path, save_file)
    with open(save_path, "w") as f:
        json.dump(samples, f)
    print(f"Generated samples saved to {save_path}")


def main():
    """Main function to run the script."""
    args = parse_arguments()
    device = setup_environment(args.seed, args.device)

    if args.trainer:
        run_trainer(args, device)
    

    if args.sampler:
        if args.prompt:
            run_conditonal_sampler(args, device)
        else:   
            run_sampler(args, device)

    if args.interact:
        run_interactive_console(args, device)

if __name__ == "__main__":
    main()