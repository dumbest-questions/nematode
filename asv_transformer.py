import numpy as np

import numpy as np

class ASVTransformer:
    def __init__(self, vocab_size, d_model=64):
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Xavier Initialization
        limit = np.sqrt(6 / (vocab_size + d_model))
        self.embeddings = np.random.uniform(-limit, limit, (vocab_size, d_model))
        
        attn_limit = np.sqrt(6 / (d_model + d_model))
        self.W_q = np.random.uniform(-attn_limit, attn_limit, (d_model, d_model))
        self.W_k = np.random.uniform(-attn_limit, attn_limit, (d_model, d_model))
        self.W_v = np.random.uniform(-attn_limit, attn_limit, (d_model, d_model))
        
        self.W_out = np.random.uniform(-limit, limit, (d_model, vocab_size))
        
        # Layer Norm params
        self.ln_gamma = np.ones(d_model)
        self.ln_beta = np.zeros(d_model)
        
        self.last_context = None 

    def save(self, filename):
        """Saves the brain's state to an .npz file."""
        np.savez(filename, 
                 embeddings=self.embeddings,
                 W_q=self.W_q, W_k=self.W_k, W_v=self.W_v,
                 W_out=self.W_out,
                 ln_gamma=self.ln_gamma,
                 ln_beta=self.ln_beta,
                 vocab_size=self.vocab_size,
                 d_model=self.d_model)
        print(f"Brain state saved to {filename}")

    @classmethod
    def load(cls, filename):
        """Creates a new model instance and loads saved weights."""
        data = np.load(filename)
        # extract vocab_size and d_model from the file to ensure a perfect fit
        model = cls(vocab_size=int(data['vocab_size']), d_model=int(data['d_model']))
        
        model.embeddings = data['embeddings']
        model.W_q, model.W_k, model.W_v = data['W_q'], data['W_k'], data['W_v']
        model.W_out = data['W_out']
        model.ln_gamma, model.ln_beta = data['ln_gamma'], data['ln_beta']
        
        print(f"Brain state loaded from {filename} (Vocab: {model.vocab_size}, D_MODEL: {model.d_model})")
        return model

    def layer_norm(self, x, eps=1e-5):
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return self.ln_gamma * (x - mean) / (std + eps) + self.ln_beta

    def forward(self, input_ids):
        # embeddings
        X = self.embeddings[input_ids]
        
        # QKV
        Q = np.dot(X, self.W_q)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)
        
        # masked attention
        scores = np.dot(Q, K.T) / np.sqrt(self.d_model)
        mask = np.triu(np.full_like(scores, -np.inf), k=1)
        
        # stable softmax
        shift_scores = (scores + mask) - np.max(scores + mask, axis=-1, keepdims=True)
        attn_weights = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis=-1, keepdims=True)
        
        # contextualize + layer norm ("grown-up" stability step)
        context_raw = np.dot(attn_weights, V)
        
        # residual context + layer norm
        self.last_context = self.layer_norm(context_raw + X) 
        
        logits = np.dot(self.last_context, self.W_out)
        return logits