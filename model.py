import torch
import torch.nn as neuralnetwork
from torch.nn import functional as functional

from dataclasses import dataclass



@dataclass
class Config:
    blockSize: int = 64
    vocabSize: int = 65
    nEmbed: int = 128
    headNumber: int = 2
    layerNumber: int = 2
    dropout: float = 0.2

    

class Head(neuralnetwork.Module):

    def __init__(self, config, headSize):
        super().__init__()
        self.key = neuralnetwork.Linear(config.nEmbed, headSize, bias=False)
        self.query = neuralnetwork.Linear(config.nEmbed, headSize, bias=False)
        self.value = neuralnetwork.Linear(config.nEmbed, headSize, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.blockSize, config.blockSize)))

        self.dropout = neuralnetwork.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B,T,headsize)
        q = self.query(x) # (B,T,headsize)

        # compute self attention affinities between tokens
        weights = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, 16) @ (B, 16, T) ---> (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = functional.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        # weighted aggregation of values
        v = self.value(x)
        output = weights @ v
        return output
    
class MultiHeadAttention(neuralnetwork.Module):

    def __init__(self, config, headSize):
        super().__init__()
        self.heads = neuralnetwork.ModuleList([Head(config, headSize) for _ in range(config.headNumber)])
        self.projection = neuralnetwork.Linear(config.nEmbed, config.nEmbed)
        self.dropout = neuralnetwork.Dropout(config.dropout)

    def forward(self, x):
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        output = self.dropout(self.projection(output))
        return output
    
class FeedForward(neuralnetwork.Module):

    def __init__(self, config):
        super().__init__()
        self.network = neuralnetwork.Sequential(
            neuralnetwork.Linear(config.nEmbed, 4 * config.nEmbed),
            neuralnetwork.ReLU(),
            neuralnetwork.Linear(4 * config.nEmbed, config.nEmbed), # projection layer
            neuralnetwork.Dropout(config.dropout)
        )
    
    def forward(self, x):
        return self.network(x)
    
class Block(neuralnetwork.Module):

    def __init__(self, config):
        super().__init__()
        headSize = config.nEmbed // config.headNumber
        self.selfAttention = MultiHeadAttention(config, headSize)
        self.feedForward = FeedForward(config)
        self.layerNorm1 = neuralnetwork.LayerNorm(config.nEmbed)
        self.layerNorm2 = neuralnetwork.LayerNorm(config.nEmbed)

    def forward(self, x):
        x = x + self.selfAttention(self.layerNorm1(x))
        x = x + self.feedForward(self.layerNorm2(x))
        return x
    
class GenerativePretrainedTransformer(neuralnetwork.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocabSize is not None
        assert config.blockSize is not None

        self.config = config

        # each token directly reads off the logits for the next token from a lookup table
        self.tokenEmbeddingTable = neuralnetwork.Embedding(config.vocabSize, config.nEmbed)
        self.positionEmbeddingTable = neuralnetwork.Embedding(config.blockSize, config.nEmbed)
        self.blocks = neuralnetwork.Sequential(*[Block(config) for _ in range(config.layerNumber)])
        self.layerNormFinal = neuralnetwork.LayerNorm(config.nEmbed)
        self.languageModelHead = neuralnetwork.Linear(config.nEmbed, config.vocabSize)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tokenEmbed = self.tokenEmbeddingTable(idx) # (B,T,C)
        positionEmbed = self.positionEmbeddingTable(torch.arange(T, device=device)) # (T, C)
        x = tokenEmbed + positionEmbed
        x = self.blocks(x) 
        x = self.layerNormFinal(x)
        logits = self.languageModelHead(x) # (B,T,vocabSize)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = functional.cross_entropy(logits, targets)

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, maxNewTokens):
    # idx is (B, T) array of indices in the current context
        for _ in range(maxNewTokens):
            # crop idx to the last blockSize tokens
            idx_cond = idx[:, -self.config.blockSize:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = functional.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx