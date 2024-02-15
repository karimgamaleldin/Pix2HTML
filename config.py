import torch

CONTEXT_SIZE = 48
IMAGE_SIZE = 256
BATCH_SIZE = 64
EPOCHS = 10
STEPS_PER_EPOCH = 72000
device = 'cuda ' if torch.cuda.is_available() else 'cpu'