import os
import numpy as np

import tiktoken

from halo import Halo

preparingLoading = Halo(text='Preparing data: ', spinner='line', color='white', placement='right')

def prepareTextFileData(filePath, encodeMethod=''):

    preparingLoading.start()

    with open(filePath, 'r') as file:
        data = file.read()
    
    length = len(data)
    trainData = data[:int(length*0.8)]
    validationData = data[int(length*0.8):]
    
    preparingLoading.stop()

    print('Data file read')
    print('Training data length: ', len(trainData))
    print('Validation data length: ', len(validationData))

    if encodeMethod == 'r50k_base':
        print('Encoding Method: r50k_base')

        encoding = tiktoken.get_encoding('r50k_base')
        trainIds = encoding.encode_ordinary(trainData)
        validationIds = encoding.encode_ordinary(validationData)
    else:
        print('prepareData ERROR: Encoding Method Undefined!')
        return

    trainIds = np.array(trainIds, dtype=np.uint16)
    validationIds = np.array(validationIds, dtype=np.uint16)

    fileName = os.path.basename(filePath)
    trainBin = os.path.join(os.path.dirname(filePath), os.path.splitext(fileName)[0]+'train.bin')
    validationBin = os.path.join(os.path.dirname(filePath), os.path.splitext(fileName)[0]+'validation.bin')

    trainIds.tofile(trainBin)
    validationIds.tofile(validationBin)

    print('Training data encoded to bin at', trainBin)
    print('Validation data encoded to bin at', validationBin)