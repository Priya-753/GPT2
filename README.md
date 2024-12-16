# GPT-2 Model Implementation from Scratch

This project demonstrates how to implement the GPT-2 model (124M) from scratch using PyTorch. The script loads pre-trained weights released by OpenAI, configures the model, and generates text based on a provided prompt.

## Requirements

- Python 3.x
- PyTorch (CUDA/MPS support recommended)
- `gpt_download3`
- `LLMArchitecture` (custom modules)
- `DataPreprocessing` (custom modules)

### Install Dependencies

Ensure all required dependencies are installed in your environment:

```bash
pip install -r requirements.txt
```

## Run

The model with the GPT-2 weights can be run by the command:
```bash
python run.py
```
The input can be modified in the run.py to see different outputs.
