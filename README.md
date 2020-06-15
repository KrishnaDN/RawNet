# RawNet
Implementation of the paper ": Advanced end-to-end deep neural network using raw waveforms for text-independent speaker verification"
Paper: https://arxiv.org/pdf/1904.08104.pdf

# Usage

```
import torch
from model import RawNet
inputs = torch.rand(64,1,59049) # Input shape (batch_size,channel_dim,no_samples)
model = RawNet(input_channel=1, num_classes=1211)
predictions, speaker_embeddings = model(inputs)
```