import os
import numpy as np
import time

import torch
import torch.nn as neuralnetwork
from torch.nn import functional as functional

import tiktoken
from model import GenerativePretrainedTransformer, Config
from prepare import prepareData

from halo import Halo

# hyperparameters #


# data
batchSize = 32 # how many independent sequences will we process in parallel?
blockSize = 64 # what is the maximum context length for predictions?
dataset = 'input.txt'
encodeMethod = 'default'

# model
nEmbed = 128
headNumber = 2
layerNumber = 2
dropout = 0.2

init = 'resume'
saveCheckpoints = True
modelName = 'minishakespeareGPT'

# training
maxIterations = 50000
evaluationInterval = 500
evaluationIterations = 200
learningRate = 3e-4


# statics
outDir = 'out'
resumeDir = 'resume'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
###################



# data loader #
dataDir = os.path.join('data', dataset)

if encodeMethod == 'r50k_base':

    encoding = tiktoken.get_encoding('r50k_base')
    vocabSize = 50304

    fileName = os.path.basename(dataDir)
    trainBin = os.path.join(os.path.dirname(dataDir), os.path.splitext(fileName)[0]+'Train.bin')
    validationBin = os.path.join(os.path.dirname(dataDir), os.path.splitext(fileName)[0]+'Validation.bin')

    if not os.path.exists(trainBin) or not os.path.exists(validationBin):
        prepareData(dataDir, 'r50k_base')

    trainData = np.memmap(os.path.join(os.path.dirname(dataDir), os.path.splitext(fileName)[0]+'Train.bin'), dtype=np.uint16, mode='r')
    validationData = np.memmap(os.path.join(os.path.dirname(dataDir), os.path.splitext(fileName)[0]+'Validation.bin'), dtype=np.uint16, mode='r')

    encoded = True

else:

    with open(dataDir, 'r', encoding='utf-8') as file:
        text = file.read()

    # assigning characters in vocab set integers for encoding and decoding
    vocab = sorted(list(set(text)))
    vocabSize = len(vocab)

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

    # encode shakespeare dataset to training tensor and validation tensor
    data = torch.tensor(encoding.encode(text), dtype=torch.long)
    n = int(0.8*len(data))
    trainData = data[:n]
    validationData = data[n:]

    encoded = False

# # # # # # # # # # # # # # # # 



# batching
def getBatch(split):
    data = trainData if split == 'train' else validationData
    randomOffsets = torch.randint(len(data) - blockSize, (batchSize,))

    if encoded:
        x = torch.stack([torch.from_numpy((data[i:i+blockSize]).astype(np.int64)) for i in randomOffsets])
        y = torch.stack([torch.from_numpy((data[i+1:i+blockSize+1]).astype(np.int64)) for i in randomOffsets])
    else:
        x = torch.stack([data[i:i+blockSize] for i in randomOffsets])
        y = torch.stack([data[i+1:i+blockSize+1] for i in randomOffsets])

    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y

# model init #
modelArgs = dict(
    blockSize = blockSize,
    vocabSize = vocabSize,
    nEmbed = nEmbed,
    headNumber = headNumber,
    layerNumber = layerNumber,
    dropout = dropout)
if init == 'new':

    print('Initializing a new model...')

    lastSteps = 0

    config = Config(**modelArgs)
    model = GenerativePretrainedTransformer(config)

    print('Model has', sum(p.numel() for p in model.parameters()), 'parameters')

elif init == 'resume':

    ckpt_path = os.path.join(resumeDir, os.listdir(resumeDir)[0])
    checkpoint = torch.load(ckpt_path, map_location=device)

    checkpointModelArgs = checkpoint['modelArgs']

    print(f"Resuming training from {ckpt_path}")

    # force these config attributes to be equal otherwise we can't even resume training
    for i in ['blockSize', 'vocabSize', 'nEmbed', 'headNumber', 'layerNumber']:
        modelArgs[i] = checkpointModelArgs[i]

    config = Config(**modelArgs)
    model = GenerativePretrainedTransformer(config)

    model.load_state_dict(checkpoint['model'])

    lastSteps = checkpoint['steps']
    print(f'from step {lastSteps}')

    print('Model has', sum(p.numel() for p in model.parameters()), 'parameters')

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate)
if init == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

# # # # # # # #

# estimating mean losses per validation
@torch.no_grad()
def estimate_loss():
    output = {}
    #model.evaluate()
    for split in ['train', 'validation']:
        losses = torch.zeros(evaluationIterations)
        for k in range(evaluationIterations):
            X, Y = getBatch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        output[split] = losses.mean()
    #model.train()
    return output

## TRAINING ##
t0 = time.time()

trainingLoading = Halo(text='Training: ', spinner='line', color='white', placement='right')
trainingLoading.start()

for steps in range(maxIterations+1):

    # every evaluationInterval evaluate the loss on train and val sets
    if steps % evaluationInterval == 0:
        
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        losses = estimate_loss()

        trainingLoading.stop()
        print(f"Step {steps+lastSteps}: train loss {losses['train']:.4f}, val loss {losses['validation']:.4f}, time {round(dt, 3)}s")

        if saveCheckpoints and steps > 0:
            
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'modelArgs': modelArgs,
                'steps': steps,
                'loss': losses['validation']
            }
            print(f"Saving checkpoint to {outDir}")
            torch.save(checkpoint, os.path.join(outDir, modelName + str(steps+lastSteps) + '.pt'))

        trainingLoading.start()

    xb, yb = getBatch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

trainingLoading.succeed(text='Training: success')

# # the output
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(encoding.decode(model.generate(context, maxNewTokens=1000)[0].tolist()))