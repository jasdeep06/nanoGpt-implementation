from datasets import load_dataset
import tiktoken
import numpy as np
import torch

# Load a subset of the OpenWebText dataset using the `load_dataset` function.
# 'stas/openwebtext-10k' is a dataset identifier for Hugging Face's Datasets library.
dataset = load_dataset('stas/openwebtext-10k')

# Initialize a tokenizer for the GPT-2 model using a utility function.
# This tokenizer will be used to convert text into a sequence of token ids.
enc = tiktoken.encoding_for_model('gpt-2')

# Print the dataset structure and the first item in the 'train' set for inspection.
print(dataset)
print(dataset['train'][0])

# Split the 'train' dataset into training and validation sets.
# Only 0.05% of the data is used for validation to ensure a large training set.
split_dataset = dataset['train'].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
# Rename the 'test' split to 'val' for clarity.
split_dataset['val'] = split_dataset.pop('test')

# Define a function to process each example in the dataset.
# This function tokenizes the text and adds an end-of-text (EOT) token.
def process(example):
    ids = enc.encode_ordinary(example['text'])  # Tokenize the text.
    ids.append(enc.eot_token)  # Append the EOT token to the tokenized list.
    out = {'ids': ids, 'len': len(ids)}  # Prepare the output dictionary.
    return out

# Apply the `process` function to each example in both the training and validation sets.
# This tokenizes all the texts and removes the original 'text' column.
tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing...",
)

# Print the first processed (tokenized) item in the 'train' set for inspection.
print(tokenized['train'][0])

# Iterate over each split in the tokenized dataset.
for split, dset in tokenized.items():
    # Sum up the lengths of all tokenized texts to determine the size of the memory-mapped array.
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    # Prepare the filename for storing the tokenized data.
    filename = split + ".bin"
    # Create a memory-mapped array with the calculated size.
    arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len,))
    # Determine the number of batches to process based on the split.
    total_batches = 5 if split == 'val' else 1024

    # Initialize an index to keep track of the current position in the memory-mapped array.
    idx = 0
    # Process each batch of data.
    for batch_idx in range(total_batches):
        print(batch_idx)
        # Shard the dataset into smaller, contiguous parts and convert to NumPy format for efficient processing.
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        # Concatenate the token ids from the batch into a single array.
        arr_batch = np.concatenate(batch['ids'])
        # Write the concatenated batch to the memory-mapped array.
        arr[idx:idx + len(arr_batch)] = arr_batch
        # Update the index for the next batch.
        idx += len(arr_batch)
    
    # Flush changes to the memory-mapped array to disk, ensuring all data is saved.
    arr.flush()


def load_dataset():
    train_arr = np.memmap('train.bin',dtype=np.uint16,mode='r')
    val_arr = np.memmap('val.bin',dtype=np.uint16,mode='r')
    # print(train_arr[0:100])
    # print(len(train_arr))
    # print(val_arr[0:100])
    # print(len(val_arr))
    return train_arr,val_arr


def get_batch(split,batch_size,block_size):
    train_arr,val_arr = load_dataset()
    arr = train_arr if split == 'train' else val_arr
    len_ds = len(arr)
    #+1 to counter range error while indexing y when len_ds-block_size is selected
    leading_indices = np.random.randint(0,len_ds-(block_size+1),(batch_size,))
    #[1,2,3,4] => [[1],[2],[3],[4]]
    leading_indices = leading_indices.reshape(batch_size,1)
    #[[1],[2],[3],[4]] => [[1,2,3],[2,3,4],[3,4,5],[4,5,6]] for block_size= 2
    batch_range_X = leading_indices + np.arange(block_size)
    #[[1,2,3],[2,3,4],[3,4,5],[4,5,6]] => [[2,3,4],[3,4,5],[4,5,6],[5,6,7]]
    batch_range_Y = batch_range_X + 1
    return torch.tensor(arr[batch_range_X].astype(np.int32)), torch.tensor(arr[batch_range_Y].astype(np.int32))

X,y = get_batch('train',8,1024)
print(X.shape,y.shape)


