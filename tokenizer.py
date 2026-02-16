import numpy as np

import re
import pickle
import collections

from tqdm import tqdm

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = {}  # (token1, token2) -> merged_token
        self.vocab = {}   # id -> bytes/string
        self.word_freqs = collections.defaultdict(int)

    def train(self, text):
        words = text.split()
        for word in words:
            # We represent words as tuples of characters: ('i', 'n', '</w>')
            char_tuple = tuple(list(word)) + ('</w>',)
            self.word_freqs[char_tuple] += 1

        unique_chars = set()
        for word_tuple in self.word_freqs:
            for char in word_tuple:
                unique_chars.add(char)
        
        self.vocab = {i: char for i, char in enumerate(sorted(list(unique_chars)))}
        self.reverse_vocab = {char: i for i, char in self.vocab.items()}
        
        num_merges = self.vocab_size - len(self.vocab)
        for i in tqdm(range(num_merges), desc="Tokenizer Training"):
            pairs = self._get_stats()
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            
            # Create a new token by concatenating the pair
            new_token = "".join(best_pair)
            new_id = len(self.vocab)
            
            self.merges[best_pair] = new_token
            self.vocab[new_id] = new_token
            self.reverse_vocab[new_token] = new_id
            
            self._merge_word_freqs(best_pair)
            
    def _get_stats(self):
        counts = collections.defaultdict(int)
        for word_tuple, freq in self.word_freqs.items():
            for i in range(len(word_tuple) - 1):
                counts[word_tuple[i:i+2]] += freq
        return counts

    def _merge_word_freqs(self, pair):
        new_word_freqs = collections.defaultdict(int)
        for word_tuple, freq in self.word_freqs.items():
            new_word = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i:i+2] == pair:
                    new_word.append("".join(pair))
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        self.word_freqs = new_word_freqs

    def encode(self, text):
        result = []
        for word in text.split():
            word_parts = list(word) + ['</w>']
            for pair, merged in self.merges.items():
                i = 0
                while i < len(word_parts) - 1:
                    if (word_parts[i], word_parts[i+1]) == pair:
                        word_parts[i:i+2] = [merged]
                    else:
                        i += 1
            result.extend([self.reverse_vocab.get(token, 0) for token in word_parts])
        return result

    def decode(self, ids):
        text = "".join([self.vocab.get(i, "[UNK]") for i in ids])
        return text.replace('</w>', ' ')

    def save(self, filename="tokenizer.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump({
                'vocab_size': self.vocab_size,
                'merges': self.merges,
                'vocab': self.vocab,
                'reverse_vocab': self.reverse_vocab
            }, f)
        print(f"Tokenizer saved to {filename}")

    @classmethod
    def load(cls, filename="tokenizer.pkl"):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        obj = cls(vocab_size=data['vocab_size'])
        obj.merges = data['merges']
        obj.vocab = data['vocab']
        obj.reverse_vocab = data['reverse_vocab']
        return obj

