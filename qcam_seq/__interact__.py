import torch
import argparse
import os
import sys
import json
import tomllib
import numpy as np
from .circuit import Circuit

def tokenize_with_max_matching(text, vocab):
    vocab_keys = set(str(k) for k in vocab.keys())
    max_len = 0
    for key in vocab_keys:
        if key:
            max_len = max(max_len, len(key))
    if max_len == 0:
        return [text] # 词汇表为空或只有空字符串，无法分割
    # 2. 初始化
    result, result_idx = [], []
    i = 0  # 当前处理的起始位置
    text_len = len(text)

    # 3. 循环匹配
    while i < text_len:
        found_match = False
        
        # 尝试从最长词条长度开始，递减查找
        # 长度 l 的范围是 [max_len, 1]
        for length in range(min(max_len, text_len - i), 0, -1):
            
            # 提取子字符串
            sub_str = text[i:i + length]
            
            # 检查词汇表
            if sub_str in vocab_keys:
                # 4. 找到匹配
                result.append(sub_str)
                result_idx.append(vocab[sub_str])
                i += length  # 移动到新的起始位置
                found_match = True
                break # 找到最长匹配后，跳出当前长度循环
                
        # 5. 处理未匹配的字符 (作为单个字符处理)
        # 理论上，由于你的词汇表包含了所有单字符 'a'-'z', ' ', '!', '?' 等，
        # 这一步在你的场景下不应该发生，但为了鲁棒性还是加上。
        if not found_match:
            # 如果最长到最短都没有匹配，则将当前字符作为单独的词条
            single_char = text[i]
            result.append(single_char)
            result_idx.append(vocab[single_char])
            i += 1
            print(f"Warning: Character '{single_char}' at position {i-1} not found in vocab.")
    return result, result_idx, len(result)





    # --- data processing functions ---

def int_to_binary_array(n, length=5):
    return np.array(list(np.binary_repr(n).zfill(length))).astype(np.int8)

# def padding(sentence_binary, max_length=100):
#     '''padding a single sentence to max_length'''
#     current_length = sentence_binary.shape[0]
#     if current_length < max_length:
#         padding_length = max_length - current_length
#         padding_array = np.zeros((padding_length, 5), dtype=np.int8)
#         padded_sentence = np.vstack((sentence_binary, padding_array))
#     else:
#         padded_sentence = sentence_binary[:max_length]
#     return padded_sentence
def parse_single_sentence(vocab, sentence, is_token=False, qubit_length=6):
    """Parse a single sentence into a sequence of binary arrays."""
    if is_token:
        _, sentence_idx, _ = tokenize_with_max_matching(sentence, vocab['char_to_token'])
        sentence = [vocab['token_to_char'][str(idx)] for idx in sentence_idx]
    print(sentence)
    tokens_binary = []
    for char in sentence:
        if char in vocab['char_to_token']:
            token = vocab['char_to_token'][char]
            binary_array = int_to_binary_array(token, length=qubit_length)
            tokens_binary.append(binary_array)
        else:
            raise ValueError(f"Character '{char}' not in vocabulary.")
    # Append the 'end' token at the end of the sentence
    # end_token = vocab['char_to_token']['end']
    # tokens_binary.append(int_to_binary_array(end_token))
    # return np.array(tokens_binary)
    return torch.tensor(np.array(tokens_binary))
# --- trans the binary array to text ---
def binseq_to_int(binseq):
    """Convert a binary sequence (list or array) to an integer."""
    return int("".join(map(str, binseq)), 2)

def _to_words(int, vocab):
    char = vocab['token_to_char'][str(int)]
    return char

def _single_samples_to_text(samples, vocab, prompt_length=0):
    text = ""

    # for sample in samples:
    binary_length = samples.shape[1]
    for i in range(prompt_length, samples.shape[0]):
        int_sample = binseq_to_int(samples[i])
          # Debugging line to check the integer value

        word = _to_words(int_sample, vocab)
        print(int_sample, samples[i], word)
        if word == "end" or word == "[END]":
            break
        text += word
    return text

# --- command functions ---
def model_info(config, model, device):
    print(f"""        
    Data Name: {config['name']}
    Length: {config['length']}
    Number of Qubits: {config['num_qubits']}
    Depth: {config['depth']}
    Reset: {config['reset']}
    Projection: {config['proj']}
    Gate: {config['gate']}
    Device: {device}
    Model Parameters: {model.efficient_parameters()}
    """)
def model_params(model):
    return model.efficient_parameters()
def help():
    print("""
    Available commands:
    $help               - Show this help message
    $model               - Display model information
    $params             - Show number of model parameters
    $exit               - Exit the interactive console
    """)
def generate_one_sample(model, vocab):
    # generate one sample
    with torch.no_grad():
        sample = model.sample(num_samples=1).cpu().numpy()
    generated_text = _single_samples_to_text(sample[0], vocab)
    print(f"[Return]: {generated_text}")


def run_interactive_console(args: argparse.Namespace, device: torch.device):
    assert args.path, "Please provide the path to the trained model using --path"
    if not os.path.exists(args.path):
        sys.exit(f"Error: Model directory not found at '{args.path}'")
    
    print("\n--- Interactive Mode ---")
    print("Enter text to process, or type a command starting with '$'.")
    print("Type '$help' for a list of commands.")
    
    with open(os.path.join(args.path, 'info.toml'), "rb") as f:
        config = tomllib.load(f)['model']
    model = Circuit(
        length=config['length'], num_qubits=config['num_qubits'], mydevice=device,
        mydtype=torch.float32, depth=config['depth'], reset=config['reset'],
        proj=config['proj'], gate=config['gate']
    ).to(device)

    model.load_state_dict(torch.load(os.path.join(args.path, 'model.pth'), map_location=device))
    model.eval()
    vocab = json.load(open(os.path.join(args.path, 'vocab.json'), 'r'))

    ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyz .,?'")
    while True:
        try:
            user_input = input("\n>> ").strip()
            if not user_input:
                continue
            
            if user_input.startswith('$'):
                command = user_input[1:].strip().lower()
                if command == "help":
                    help()
                elif command == "model":
                    model_info(config, model, device)
                elif command == "params":
                    print(f"Model Parameters: {model_params(model)}")
                elif command == "exit":
                    exit()
                elif command == "g" or command == "gen" or command == "generate":
                    generate_one_sample(model, vocab)
                else:
                    print(f"Unknown command: {command}. Type '$help' for a list of commands.")
            else:
                user_input = user_input.lower()
                if any(char not in ALLOWED_CHARS for char in user_input):
                    print("Error: Input contains invalid characters. Allowed characters are:")
                    print("abcdefghijklmnopqrstuvwxyz .,?'")
                    continue
                
                try:
                    input_data = parse_single_sentence(vocab, user_input).to(device)
                except ValueError as e:
                    print(e)
                    continue
                
                with torch.no_grad():
                    generated_samples = model.cond_sample(num_samples=1, prompt=input_data).cpu().numpy()
                    # print(generated_samples)
                    generated_text = _single_samples_to_text(generated_samples, vocab)
                    print(f"[Return]: {generated_text}")
        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break
        except Exception as e:
            print(f"Error: {e}")