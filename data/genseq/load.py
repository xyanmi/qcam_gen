


def binseq_to_int(binseq):
    """Convert a binary sequence (list or array) to an integer."""
    return int("".join(map(str, binseq)), 2)

def _to_words(int, vocab):
    char = vocab['token_to_char'][str(int)]
    return char

def _single_samples_to_text(samples, vocab):
    text = ""
    for sample in samples:
        int_sample = binseq_to_int(sample)
        word = _to_words(int_sample, vocab)
        if word == "end":
            break
        text += word
    return text

def samples_to_text(samples, vocab):
    texts = []
    for sample in samples:
        text = _single_samples_to_text(sample, vocab)
        texts.append(text)
    return texts


import json
with open("./vocab.json", "r") as f:
    vocab = json.load(f)

import numpy as np
samples = np.load("./binary_sentences.npy")[:,:,0:5]
print(samples.shape)
texts = samples_to_text(samples, vocab=vocab)
with open("decoded_sentences.txt", "w") as f:
    for text in texts:
        f.write(text + "\n")