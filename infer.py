import numpy as np
import string
from tokenizer import BPETokenizer
from asv_transformer import ASVTransformer

MODEL_FILENAME = "asv_bible_brain.npz"
TOKENIZER_FILENAME = "bible_tokenizer.pkl"

D_MODEL = 64

def generate_top_k(model, tokenizer, start_phrase, length=20, temperature=0.7, k=5):
    # (Your existing generate_top_k logic goes here)
    words = start_phrase.lower().split()
    ids = tokenizer.encode(" ".join(words))
    for _ in range(length):
        logits = model.forward(ids)[-1, :] / temperature
        top_k_indices = np.argpartition(logits, -k)[-k:]
        top_k_logits = logits[top_k_indices]
        shift_logits = top_k_logits - np.max(top_k_logits)
        probs = np.exp(shift_logits) / np.sum(np.exp(shift_logits))
        next_id = np.random.choice(top_k_indices, p=probs)
        ids.append(next_id)
    return tokenizer.decode(ids)

tokenizer = BPETokenizer.load(TOKENIZER_FILENAME)
brain = ASVTransformer.load(MODEL_FILENAME)

test_prompts = {
    "Biblical Start": "In the beginning",
    "Standard Start": "The implementation shall",
    "Template Prophecy": "template <typename T> class Prophet",
    "Commandment Swap": "Thou shalt not use",
    "Theological Hybrid": "The pointer of the Lord",
    "Direct Collision": "And God said, let there be a",
    "Undefined Sin": "It is undefined behavior to"
}

# temperatures from [0.7 to 0.8] for max creativity, but we can also try 0.6 for a more 'focused' output. 
# C++ is very rigid, so a higher temp helps it "break character" into the Bible.

print(f"\n{'='*60}\B++ MODEL EVALUATION\n{'='*60}")

for category, prompt in test_prompts.items():
    print(f"\n[Category: {category}]")
    print(f"Prompt: \"{prompt}\"")
    
    # Generate 3 variations for each to see the 'Stochastic Range'
    for i in range(5):
        # We use a slightly lower temperature to see the 'core' logic
        result = generate_top_k(brain, tokenizer, prompt, length=20, temperature=0.6, k=5)
        print(f"  Result {i+1}: ...{result}")