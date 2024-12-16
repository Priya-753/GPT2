import torch
from LLMArchitecture.gpt_download3 import download_and_load_gpt2
from LLMArchitecture.gpt_model import GPTModel
from DataPreprocessing.tokenizer import GPT2Tokenizer
from LLMArchitecture.train_utils import generate, text_to_token_ids, token_ids_to_text, load_weights_into_gpt

# GPT-2 Configuration for 124M model
GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (original: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

# Initialize tokenizer
tokenizer = GPT2Tokenizer()

# Download GPT-2 parameters and settings
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())

# Define configurations for different GPT-2 variants
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Select model variant and update base configuration
model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

# Initialize the GPT-2 model
gpt = GPTModel(NEW_CONFIG)
gpt.eval()

# Configure device (GPU, MPS, or CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Load pre-trained weights into the model
load_weights_into_gpt(gpt, params)
gpt.to(device)

# Set manual seed for reproducibility
torch.manual_seed(123)

# Input prompt for text generation
input_text = "Every effort moves you"
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids(input_text, tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)

# Convert generated token IDs back to text and print the output
output_text = token_ids_to_text(token_ids, tokenizer)
print("Output text:\n", output_text)
