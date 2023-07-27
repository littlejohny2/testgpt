# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np

import tiktoken
from datasets import load_dataset # huggingface datasets



# params #

datasetName = 'wikitext'
validationSize = 0.005
encodeMethod = 'r50k_base'

##########



if __name__ == '__main__':

    dataset = load_dataset(datasetName, 'wikitext-103-v1') # params

    print(dataset)


    splitDataset = dataset['train'].train_test_split(test_size=validationSize, seed=2357, shuffle=True)
    splitDataset['validation'] = splitDataset.pop('test')


    # tokenization
    encoding = tiktoken.get_encoding(encodeMethod)
    def tokenization(example):
        ids = encoding.encode_ordinary(example['text'])
        out = {'ids': ids, 'len': len(ids)}
        return out
    
    tokenized = splitDataset.map(
        tokenization,
        remove_columns=['text'],
        desc="tokenizing the splits"
    )
    

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{datasetName+split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()