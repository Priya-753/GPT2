import re
import tiktoken

class BaseTokenizer:
    def __init__(self, vocab=None):
        if vocab is not None:
            self.str_to_int = vocab
            self.int_to_str = {i: s for s, i in vocab.items()}
        else:
            self.str_to_int = None
            self.int_to_str = None

    def encode(self, text):
        raise NotImplementedError("Subclasses must implement the `encode` method")

    def decode(self, ids):
        raise NotImplementedError("Subclasses must implement the `decode` method")


class SimpleTokenizerV1(BaseTokenizer):
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


class SimpleTokenizerV2(BaseTokenizer):
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


class GPT2Tokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

# "gpt2": tiktoken.get_encoding("gpt2"),
# "gpt3": tiktoken.get_encoding("p50k_base"),  # Commonly associated with GPT-3 models
# "gpt4": tiktoken.get_encoding("cl100k_base")  # Used for GPT-4 and later versions
