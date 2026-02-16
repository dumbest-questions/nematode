import numpy as np
import string
from tokenizer import BPETokenizer
from asv_transformer import ASVTransformer
from trainer import StochasticTrainer

MODEL_FILENAME = "asv_bible_brain.npz"
TOKENIZER_FILENAME = "bible_tokenizer.pkl"
D_MODEL = 64
ITERATIONS = 50000
CONTEXT = 12

with open('bible.txt', 'r') as f:
    bible_text = f.read() 

with open('cpp_draft.txt', 'r') as f:
    cpp_draft = f.read() 

training_text = cpp_draft + "\n" + bible_text
tokenizer = BPETokenizer(vocab_size=5000)
tokenizer.train(training_text)
tokenizer.save(TOKENIZER_FILENAME)

brain = ASVTransformer(tokenizer.vocab_size, d_model=D_MODEL)
trainer = StochasticTrainer(brain, tokenizer, learning_rate=0.005)

print(f"Vocab size: {tokenizer.vocab_size} | Tokens: {len(tokenizer.encode(training_text))}")

trainer.train(training_text, iterations=ITERATIONS, context_window=CONTEXT)
brain.save(MODEL_FILENAME)
