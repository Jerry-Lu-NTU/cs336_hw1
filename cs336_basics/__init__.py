import importlib.metadata
from . import checkpoint, data, model, nn, optim, tokenizer

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass
