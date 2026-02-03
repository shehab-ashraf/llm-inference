from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self, path):
        self.t = AutoTokenizer.from_pretrained(path, legacy=False)
        self.n_words = self.t.vocab_size
        self.bos_id = self.t.bos_token_id
        self.eos_id = self.t.eos_token_id

    def encode(self, s, bos, eos):
        t = self.t.encode(s, add_special_tokens=False)
        if bos and self.bos_id is not None: t = [self.bos_id] + t
        if eos and self.eos_id is not None: t = t + [self.eos_id]
        return t

    def decode(self, t):
        return self.t.decode(t)

    def decode_token(self, t):
        # Llama tokenizers often strip spaces if decoding a single token.
        # We decode with a dummy token and take the difference.
        res = self.t.decode([self.t.bos_token_id, t])
        bos = self.t.decode([self.t.bos_token_id])
        return res[len(bos):]
