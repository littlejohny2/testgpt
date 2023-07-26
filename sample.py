import os
import numpy as np

import torch
import torch.nn as neuralnetwork
from torch.nn import functional as functional

import tiktoken
from model import GenerativePretrainedTransformer, Config

from halo import Halo



# hyperparameters #

encodeMethod = 'r50k_base'
dataset = '' # required if encode method isnt from tiktoken
samples = 8
maxNewTokens = 64 # number of tokens per sample
printConsole = False
writeToFile = True

resumeDir = 'resume'
outDir = 'out'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
###################



# encoding chooser #
dataDir = os.path.join('data', dataset)

if encodeMethod == 'r50k_base':

    encoding = tiktoken.get_encoding('r50k_base')

else:

    with open(dataDir, 'r', encoding='utf-8') as file:
        text = file.read()

    # assigning characters in vocab set integers for encoding and decoding
    vocab = sorted(list(set(text)))

    stringToInteger = { character:integer for integer, character in enumerate(vocab) }
    integerToString = { integer:character for integer, character in enumerate(vocab) }
    encode = lambda string: [stringToInteger[c] for c in string]
    decode = lambda integerList: ''.join([integerToString[i] for i in integerList])

    class Encoding:

        def encode(text: str) -> list[int]:
            integerList = encode(text)
            return integerList
        
        def decode(integerList: list[int]) -> str:
            string = decode(integerList)
            return string
        
    encoding = Encoding
# # # # # # # # # # # #


# model init
ckpt_path = os.path.join(resumeDir, os.listdir(resumeDir)[0])
checkpoint = torch.load(ckpt_path, map_location=device)

print(f"Loading model from {ckpt_path}")

config = Config(**checkpoint['modelArgs'])
model = GenerativePretrainedTransformer(config)

model.load_state_dict(checkpoint['model'])

model.to(device)

print('Model has', sum(p.numel() for p in model.parameters()), 'parameters')

# sampling
samplingLoading = Halo(text='Sampling: ', spinner='line', color='white', placement='right')
samplingLoading.start()

context = torch.zeros((1, 1), dtype=torch.long, device=device)

with torch.no_grad():
    for sample in range(samples):

        y = encoding.decode(model.generate(context, maxNewTokens)[0].tolist())

        if printConsole:
            print(f'\n{y}')
        
        if writeToFile:
            fileName = os.path.basename(ckpt_path)
            newFileName = os.path.splitext(fileName)[0] + 'Sample' + str(sample) + '.txt'
            newFile = os.path.join(outDir, newFileName)

            open(newFile, 'w').write(y)


samplingLoading.stop()