import numpy as np
import random

from tqdm import tqdm

class StochasticTrainer:
    def __init__(self, model, tokenizer, learning_rate=0.01):
        self.model = model
        self.tokenizer = tokenizer
        self.lr = learning_rate

    def train(self, full_text, iterations=1000, context_window=10):
        all_ids = self.tokenizer.encode(full_text)

        for i in tqdm(range(iterations), desc="Stochastic Sample Training"):
            # sample a random snippet of text for training
            start = random.randint(0, len(all_ids) - context_window - 1)
            chunk = all_ids[start : start + context_window + 1]
            
            inputs = chunk[:-1]
            targets = chunk[1:]

            # get the model's predictions (logits) for the input snippet
            logits = self.model.forward(inputs)
            
            # backprop - calculate the error for each word in the snippet
            for j in range(len(inputs)):
                exp_logits = np.exp(logits[j] - np.max(logits[j]))
                probs = exp_logits / np.sum(exp_logits)
                
                error_signal = probs.copy()
                error_signal[targets[j]] -= 1  # Subtract 1 from the correct word's prob
                
                self.model.W_out -= np.outer(self.model.last_context[j], error_signal) * self.lr


