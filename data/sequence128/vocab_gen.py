import json
import string 

special_tokens = {' ': 0, ',': 27, '.': 28, '?': 29, '!': 30, 'endl': 31}
chars = string.ascii_lowercase
char_to_token = {}
char_to_token.update(special_tokens)

for i, char in enumerate(chars):
    char_to_token[char] = i + 1

token_to_char = {token_id: char for char, token_id in char_to_token.items()}

vocab = {
    'char_to_token': char_to_token,
    'token_to_char': token_to_char
}

filepath = 'vocab.json'
with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(vocab, f, indent=2)

print(f"save_path: {filepath}")
print("char_to_token:", list(char_to_token.items())[:5]) 
print("token_to_char:", list(token_to_char.items())[:5])