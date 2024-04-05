# Transformers

An effort to implement a variant of GPT-2 from scratch using Pytorch.

Configs - block size(1024), vocab size(50304), number_of_transformer_blocks(12), number_of_transformer_heads(12), dimensionality_of_embeddings(768), dropout(0), bias_in_linear_or_layernorms(True)

## Result

Trained on openwebtext-100k(https://huggingface.co/datasets/Elriggs/openwebtext-100k)

Training curve


![tr-curve](https://i.ibb.co/HhnVRcR/W-B-Chart-4-6-2024-1-24-54-AM.png)


## Conclusion

Successfully overfitted on training data, proving the model was implemented correctly.



## Explaination(self-note)



1. X and y (inputs and targets) have dimensions of (batch_size, block_size). 

    Here each batch of X contains list of indexes with length of list as the context size/block size.


    Eg. tokenized sentence -> [My,name,is,jasdeep] => Looking up index in the vocabulary => [1000,1050,4,23]


    Y contains the subsequent output [name,is,jasdeep,.] => Looking up index in the vocabulary => [1050,4,23,8]


    	

2. The first layer X passes through is the embedding layer(vocab_size,embeddding_dimension). The embedding layer takes in the corresponding indices, plucks the embedding of an index and spits out. The transformation is from 

    (batch_size,block_size)  => (batch_size,block_size,embeddding_dimension)

3. The second layer is the positional embeddings.A trainable embedding of shape (block_size,embedding_dimension) is simply added to X. The shape of X remains the same (batch_size,block_size,embeddding_dimension). Positional embedding can be visualized as numbers that inform the model about the position of tokens in the input. 
4.  Dropout is applied as per the configuration.
5. This is where the first attention block starts.
    1. Layer norm is applied => The way layer norm works is as follows -
        1. Let’s say the input dimension is [1,1024,768] i.e. num_tokens = 1024,n_dim=768 and bs=1. 
        2. We’ll take every row independently i.e. in total 1024 rows. For each row we’ll find the mean and variance of the 768 n_embed and normalize that row with it.
        3. Once all the rows are independently normalized, we’ll multiply the normalized matrix with a weight and add a bias. This weight will have 768 trainable parameters that will be learnt. We are essentially asking the question, if not normalization, then what?
    2. The input is first upscaled to get 3 matrices k,q and v. It is passed through a linear layer with weights dimension (embedding_dimension,3Xembedding dimension). The shape of the resulting tensor is (batch_size,block_size,3Xembedding_dimension).
    3. It is then split into 3 tensors to give k,q and v , each of dimension (batch_size,block_size,embedding_dimension)
    4. k, q and v are then reshaped to (batch_size,block_size,num_heads,embedding_dimension//num_heads). For example if the input was [1,8,64] and num_heads = 4, the reshaped tensor would be [1,8,4,16]. This can be visualized as 8 sheets of paper parallel to each other each having matrices with 4 rows and 16 columns. The initial setup had 1 row and 64 columns. These 64 are the embeddings of the particular token represented with the sheet here.These 64 embeddings were now split into 4 rows of 16 elements each.
    5. We need to perform multihead attention on 16 embedding elements of all tokens combined on one attention head. Thus 4 heads will cover all 4*16 embedding elements. To do this, we need the dimension after the batch size be 4, so we take transpose of the k,q,v vectors to make their dimension as [1,4,8,16] or (batch_size,num_heads,block_size,embedding_dimension//num_heads). This can be visualized as grouping 16 embedding elements in each head for all the tokens.
    6. Now attention is applied. k is transposed to get the dimension (batch_size,num_heads,embedding_dimension//num_heads,block_size) and is multiplied with q to get attention scores of dimension (batch_size,num_heads,block_size,block_size). This gives the strength of token interactions between words in a block.
    7. We need to make sure that attention scores are only accounted for tokens preceding a given token, i.e. masked attention. For this, the attention scores are added to a unit lower triangular matrix with zeros replaced with infinity. After the addition, softmax is applied to get normalized attention scores. This gives us attention scores retained only for the lower triangular part while the upper triangular had become 0. The shape remains (batch_size,num_heads,block_size,block_size).
    8. In order to get the attention rich embeddings, the attention scores are multiplied with the v vector. (batch_size,num_heads,block_size,block_size) * (batch_size,num_heads,block_size,embedding_dimensions//num_heads) => (batch_size,num_heads,block_size,embedding_dimension//num_heads)
    9. Now the output is transposed back to block_size as the reference and reshaped to give the dimension (batch_size,block_size,embedding_dimension)
    10. Now some compute is added i.e. a linear layer of size (embedding_dimension,embedding_dimension) to get the final output  (batch_size,block_size,embedding_dimension)
    11. This output out of the attention block is added to the input to the attention block, much like “skip connection”
    12. Layer norm is applied.
    13. MLP is applied with first linear layer with dimension (embedding_dimension,4* embedding_dimension) followed by GELU followed by another linear layer with dimension (4*embedding_dimension,embedding dimension) and finally dropout.
    14. The output of the MLP is of dimension (batch_size,block_size,embedding_dimension) which is added back to the output of k. point(pre layer norm) as a “skip connection” to generate the final output.
6. 12 such transformer blocks are applied.
7. On the output of last transformer block, layer norm is applied
8. Finally a linear layer of shape (embedding_dimension,vocab_size) is applied to lead final distribution (batch_size,block_size,vocab_size)