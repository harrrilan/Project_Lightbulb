import sys
import torch
import transformers
import numpy
import platform
import chromadb

print("Python version:", sys.version)
print("Platform:", platform.platform())
print("Torch version:", torch.__version__)
print("Transformers version:", transformers.__version__)
print("Numpy version:", numpy.__version__)
print("Chroma version:", chromadb.__version__)

# If you want to check the tokenizer version specifically:
from transformers import BertTokenizer
print("BertTokenizer version:", BertTokenizer.__module__)