import json
import numpy as np

# the function to transform the int number to binary array
def int_to_binary_array(n, length=5):
    """Convert an integer to a binary array of a given length."""
    return np.array(list(np.binary_repr(n).zfill(length))).astype(np.int8)

# open vocalbulary file
with open("./vocab.json", "r") as f:
    vocab = json.load(f)
# open sentence file
with open("./sentence.json", "r") as f:
    sentences = json.load(f)

def padding(sentence_binary, max_length=36):
    '''padding a single sentence to max_length'''
    current_length = sentence_binary.shape[0]
    if current_length < max_length:
        padding_length = max_length - current_length
        padding_array = np.ones((padding_length, 6), dtype=np.int8)
        padding_array[:, -1] = 0  # Set the mask to 0
        padded_sentence = np.vstack((sentence_binary, padding_array))
    else:
        padded_sentence = sentence_binary[:max_length]
    return padded_sentence

def parse_single_sentence(sentence):
    """Parse a single sentence into a sequence of binary arrays."""
    tokens_binary = []
    for char in sentence:
        if char in vocab['char_to_token']:
            token = vocab['char_to_token'][char]
            binary_array = int_to_binary_array(token)
            binary_array = np.append(binary_array, 1) # 1 is the mask for real token
            tokens_binary.append(binary_array)
        else:
            raise ValueError(f"Character '{char}' not in vocabulary.")
    # Append the 'end' token at the end of the sentence
    end_token = vocab['char_to_token']['end']
    end_token = np.append(int_to_binary_array(end_token), 1) # 1 is the mask for real token
    tokens_binary.append(end_token)
    # return np.array(tokens_binary)
    return padding(np.array(tokens_binary))


# print(parse_single_sentence("the cat is watching a fly buzz around the room, waiting for the right moment to pounce.").shape[0])
max_length = -1
data = []
for sentence in sentences:
    sentence_binary = parse_single_sentence(sentence)
    data.append(sentence_binary)
    l = len(sentence_binary) + 1  # +1 for the 'end' token
    max_length = max(max_length, l)
print(f"Max length: {max_length}, sentences: {len(sentences)}")
data = np.array(data)
np.random.seed(42)
data = np.random.permutation(data)
print(f"Data shape: {data.shape}")
np.save("binary_sentences.npy", data)
