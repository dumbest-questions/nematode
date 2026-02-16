import numpy as np
import string
from tokenizer import BPETokenizer
from asv_transformer import ASVTransformer
from trainer import StochasticTrainer

FILENAME = "asv_bible_brain.npz"
D_MODEL = 64
ITERATIONS = 50000
CONTEXT = 12

with open('bible.txt', 'r') as f:
    bible_text = f.read() #.replace('\n', ' ')

with open('cpp_draft.txt', 'r') as f:
    cpp_draft = f.read() #.replace('\n', ' ')

training_text = cpp_draft + "\n" + bible_text
tokenizer = BPETokenizer(vocab_size=5000)
tokenizer.train(training_text)
tokenizer.save("bible_tokenizer.pkl")

brain = ASVTransformer(tokenizer.vocab_size, d_model=D_MODEL)
trainer = StochasticTrainer(brain, tokenizer, learning_rate=0.005)

print(f"Vocab size: {tokenizer.vocab_size} | Tokens: {len(tokenizer.encode(training_text))}")

trainer.train(training_text, iterations=ITERATIONS, context_window=CONTEXT)
brain.save(FILENAME)
