from dataclasses import dataclass
import torch.nn as nn
import tiktoken
import torch
import torch.nn.functional as F




@dataclass
class GPTConfig:
    """
    Configuration class for the GPT model.
    
    Args:
        block_size (int): The maximum sequence length of the input.
        vocab_size (int): The size of the vocabulary.
        num_transformer_block (int): The number of transformer blocks in the model.
        num_transformer_heads (int): The number of attention heads in each transformer block.
        n_embed (int): The dimensionality of the embedding layer.
        dropout (int): The dropout rate.
        bias (bool): Whether to include bias terms in the model.
    """
    block_size: int = 1024
    vocab_size: int = 50304
    num_transformer_block: int = 12
    num_transformer_heads: int = 12
    n_embed: int = 768
    dropout: int = 0
    bias: bool = True


class LayerNorm(nn.Module):
    """
    Layer normalization module.

    Args:
        ndim (int): The number of dimensions in the input tensor.
        bias (bool): Whether to include a bias term in the normalization.

    Attributes:
        weights (nn.Parameter): Learnable parameter representing the scaling factor.
        bias (nn.Parameter or None): Learnable parameter representing the bias term, or None if bias is False.

    """

    def __init__(self, ndim, bias):
        super().__init__()
        # Initialize the weights as learnable parameters with shape (ndim)
        self.weights = nn.Parameter(torch.ones(ndim))
        # Initialize the bias as learnable parameters with shape (ndim) if bias is True, otherwise set it to None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        """
        Forward pass of the layer normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.

        """
          # Apply layer normalization to the input tensor x using the weights 
          # and bias parameters of the LayerNorm module. Use a small epsilon value of 1e-5 for numerical stability.
        return F.layer_norm(x, self.weights.shape, self.weights, self.bias, 1e-5)

    

class Block(nn.Module):
    """
    A block module in the nanoGPT model.

    Args:
        config (object): Configuration object containing model parameters.

    Attributes:
        layer_norm (LayerNorm): Layer normalization module.
        attention (CausalAttention): Causal attention module.
        mlp (MLP): Multi-layer perceptron module.

    Methods:
        forward(x): Performs forward pass through the block.

    """

    def __init__(self, config):
        super().__init__()
        # Initialize the layer normalization module with the specified number of dimensions and bias parameter
        self.layer_norm = LayerNorm(config.n_embed, config.bias)
        
        # Initialize the causal attention module with the given configuration
        self.attention = CausalAttention(config)
        
        # Initialize the layer normalization module again with the specified number of dimensions and bias parameter
        self.layer_norm = LayerNorm(config.n_embed, config.bias)
        
        # Initialize the multi-layer perceptron module with the given configuration
        self.mlp = MLP(config)
    
    def forward(self, x):
        """
        Performs forward pass through the block.

        Args:
            x (tensor): Input tensor.

        Returns:
            tensor: Output tensor after passing through the block.

        """
        # Apply attention mechanism to the input tensor x and add it to x
        x = x + self.attention(self.layer_norm(x))
        
        # Apply multi-layer perceptron to the normalized input tensor x and add it to x
        x = x + self.mlp(self.layer_norm(x))
        
        # Return the updated tensor x
        return x
    
class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module.

    Args:
        config (object): Configuration object containing model parameters.

    Attributes:
        c1 (nn.Linear): Linear layer for the first hidden layer.
        act1 (nn.GELU): Activation function for the first hidden layer.
        c2 (nn.Linear): Linear layer for the second hidden layer.
        dropout (nn.Dropout): Dropout layer for regularization.

    Methods:
        forward(x): Forward pass of the MLP.

    """

    def __init__(self, config):
        super().__init__()
        # Linear layer for the first hidden layer
        self.c1 = nn.Linear(config.n_embed, 4 * config.n_embed)
        
        # Activation function for the first hidden layer
        self.act1 = nn.GELU()
        
        # Linear layer for the second hidden layer
        self.c2 = nn.Linear(4 * config.n_embed, config.n_embed)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        # Apply linear transformation to the input
        x = self.c1(x)
        
        # Apply activation function to the output of the first hidden layer
        x = self.act1(x)
        
        # Apply linear transformation to the output of the activation function
        x = self.c2(x)
        
        # Apply dropout regularization to the output
        x = self.dropout(x)
        
        # Return the final output tensor
        return x


class CausalAttention(nn.Module):
    """
    CausalAttention module performs causal attention mechanism.

    Args:
        config (object): Configuration object containing model parameters.

    Attributes:
        input_projection (nn.Linear): Linear layer for input projection.
        bias (torch.Tensor): Buffer tensor for bias.
        output_projection (nn.Linear): Linear layer for output projection.

    Methods:
        forward(x): Performs forward pass of the CausalAttention module.

    """

    def __init__(self, config):
        super().__init__()
        # Initialize the input projection layer
        self.input_projection = nn.Linear(config.n_embed, 3 * config.n_embed)
        
        # Create a buffer tensor for the bias
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        
        # Initialize the output projection layer
        self.output_projection = nn.Linear(config.n_embed, config.n_embed)

    def forward(self, x):
            """
            Performs forward pass of the CausalAttention module.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, sequence_length, hidden_size).

            """
            # Get the dimensions of the input tensor
            b, t, c = x.size()

            # Apply input projection to the input tensor
            x = self.input_projection(x)

            # Split the input tensor into key, query, and value tensors
            k, q, v = torch.split(x, config.n_embed, dim=2)

            # Reshape and transpose the key, query, and value tensors
            k = k.view(b, t, config.num_transformer_heads, c // config.num_transformer_heads).transpose(2, 1)
            q = q.view(b, t, config.num_transformer_heads, c // config.num_transformer_heads).transpose(2, 1)
            v = v.view(b, t, config.num_transformer_heads, c // config.num_transformer_heads).transpose(2, 1)

            # Compute attention scores
            attn_scores = q @ k.transpose(-2, -1)

            # Mask attention scores
            attn_scores = attn_scores.masked_fill(self.bias[:, :, :t, :t] == 0, float('-inf'))

            # Apply softmax to compute attention weights
            attn_scores = F.softmax(attn_scores, dim=-1)

            # Compute the output tensor by multiplying attention weights with value tensor
            y = attn_scores @ v

            # Transpose and reshape the output tensor
            y = y.transpose(1, 2).contiguous().view(b, t, c)

            # Apply output projection to the output tensor
            y = self.output_projection(y)

            return y




config = GPTConfig()

class GPT(nn.Module):
    def __init__(self, config):
        """
        Initializes the GPT model.

        Args:
            config (object): Configuration object containing model parameters.

        """
         # Initialize the parent class (nn.Module)
        super().__init__() 
        
        # Embedding layer for token indices
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embed) 
        
        # Embedding layer for positional encoding
        self.pos_embedding = nn.Embedding(config.block_size, config.n_embed)  
        
        # Dropout layer for transformer
        self.trans_dropout = nn.Dropout(config.dropout)  
        
        # List of transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_transformer_block)])  
        
        # Layer normalization
        self.layer_norm = LayerNorm(config.n_embed, config.bias)  
        
        # Linear layer for language modeling head
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)  

        # Tie the weights of token embedding and language modeling head
        self.token_embedding.weight = self.lm_head.weight  

        # Apply weight initialization function to all modules in the model
        self.apply(self._init_weights)  

    def forward(self, idx, targets=None):
        """
        Performs forward pass of the GPT model.

        Args:
            idx (torch.Tensor): Input tensor of token indices.
            targets (torch.Tensor): Target tensor of token indices.

        Returns:
            torch.Tensor: Logits tensor.
            torch.Tensor: Loss tensor if targets are provided, else None.

        """
        # Get the device on which the input tensor is located
        device = idx.device  

        # Get the batch size and sequence length of the input tensor
        b, t = idx.size() 
        
        # Embed the token indices using the token embedding layer
        x = self.token_embedding(idx)  
        
        # Create a tensor of positional indices
        reference = torch.arange(0, t, dtype=torch.long, device=device)  
        
         # Embed the positional indices using the positional embedding layer
        pos = self.pos_embedding(reference) 
        
        # Apply dropout to the sum of token and positional embeddings
        x = self.trans_dropout(x + pos)  
        
        # Iterate over each transformer block
        for block in self.blocks:  
        
            # Apply the transformer block to the input tensor
            x = block(x)  
        
        # Apply layer normalization to the output tensor
        x = self.layer_norm(x)  

        
        # If targets are provided
        if targets is not None:  
        
            # Compute the logits using the language modeling head
            logits = self.lm_head(x)  
        
            # Compute the cross-entropy loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)  
        
        # If targets are not provided
        else:  
        
            # Compute the logits for the last token in the sequence
            logits = self.lm_head(x[:, [-1], :])  
        
            # Set the loss to None
            loss = None  

        
        # Return the logits and loss (if available)
        return logits, loss  

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generates new tokens using the GPT model.

        Args:
            idx (torch.Tensor): Input tensor of token indices.
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Temperature value for controlling randomness.
            top_k (int): Number of top-k tokens to consider for sampling.

        Returns:
            torch.Tensor: Tensor of generated token indices.

        """
         # Iterate for the maximum number of new tokens to generate
        for _ in range(max_new_tokens): 
            
            # Generate output using the GPT model
            output = self(idx)  
            
            # Get the relevant output for the last token and apply temperature
            relevant_output = output[:, -1, :] / temperature  
            
           # Compute the probabilities of the next token using softmax
            probs = F.softmax(relevant_output, dim=-1)  
            
             # Sample the next token based on the probabilities
            idx_next = torch.multinomial(probs, num_samples=1) 
            
            # Concatenate the next token to the input tensor
            idx = torch.cat((idx, idx_next), dim=1)  
        
        # Return the generated token indices
        return idx  

    def get_num_parameters(self, non_embedding=True):
        """
        Calculates the total number of parameters in the model.

        Args:
            non_embedding (bool): Whether to exclude embedding parameters.

        Returns:
            int: Total number of parameters.

        """
        # Initialize the variable to store the total number of parameters
        total_parameters = 0
        
        # Iterate over each parameter in the model
        for parameter in self.parameters():
            # Add the number of elements in the parameter to the total count
            total_parameters += parameter.data.numel()
        
        # If non_embedding is True, subtract the number of elements in the positional embedding
        if non_embedding:
            total_parameters -= self.pos_embedding.weight.numel()

        # Return the total number of parameters
        return total_parameters

    def _init_weights(self, module):
        """
        Initializes the weights of the model.

        Args:
            module (nn.Module): Module to initialize weights for.

        """
        if isinstance(module, nn.Linear):
            # Initialize the weights of the linear module with a normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                # Initialize the bias of the linear module with zeros
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Initialize the weights of the embedding module with a normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizer(self, weight_decay, learning_rate, betas):
        """
        Configures and returns an optimizer for the model.

        Args:
            weight_decay (float): The weight decay value for the optimizer.
            learning_rate (float): The learning rate for the optimizer.
            betas (Tuple[float, float]): Coefficients used for computing running averages of gradient and its square.

        Returns:
            torch.optim.Optimizer: The configured optimizer.

        """
        param_dict = {pn: p for pn, p in self.named_parameters()}

        # Create a dictionary of parameter names and their corresponding values
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Filter out parameters that do not require gradients
        decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]

        # Get parameters with dimensions greater than or equal to 2
        nodecay_params = [p for pn, p in param_dict.items() if p.dim() < 2]

        # Get parameters with dimensions less than 2
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Group parameters into two groups: decayed and non-decayed, with different weight decay values
        num_decay_params = sum(p.numel() for p in decay_params)

        # Calculate the total number of parameters in the decayed group
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        # Calculate the total number of parameters in the non-decayed group
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        
        # Print the number of decayed parameter tensors and their total number of parameters
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create an AdamW optimizer with the specified groups of parameters, learning rate, and betas
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

        # Return the configured optimizer
        return optimizer
    


        












# text = """
# Rafael Nadal Parera a Spanish professional tennis player. He has been ranked world No. 1 in singles by the Association of Tennis Professionals (ATP) for 209 weeks, and has finished as the year-end No. 1 five times. Nadal has won 22 Grand Slam men's singles titles, including a record 14 French Open titles. He has won 92 ATP-level singles titles, including 36 Masters titles and an Olympic gold medal, with 63 of these on clay courts. Nadal is one of only two men to complete the Career Golden Slam in singles.[b] His 81 consecutive wins on clay constitute the longest single-surface win streak in the Open Era. For over a decade, Nadal has led men's tennis along with Roger Federer and Novak Djokovic as the Big Three.[c] At the start of his professional career, Nadal became one of the most successful teenagers in ATP Tour history, reaching the world No. 2 ranking and winning 16 titles before turning 20, including his first French Open and six Masters events. Nadal became the world No. 1 for the first time in 2008 after defeating Federer in a historic Wimbledon final, his first major victory off clay. He followed up his win with an Olympic singles gold at the 2008 Beijing Olympics. After defeating Djokovic in the 2010 US Open final, then-24-year-old Nadal became the youngest man in the Open Era to achieve the Career Grand Slam, and the first man to win majors on three different surfaces (hard, grass, and clay) in the same year (Surface Slam).After two injury-plagued seasons, Nadal returned to the Tour in 2013, reaching 14 finals, winning two majors and five Masters events including the US Open Series sweep (Summer Slam). He continued his dominance at the French Open, securing six titles, two US Open titles, an Australian Open title, and an Olympic doubles gold at the 2016 Rio Olympics with Marc López. Nadal surpassed his joint-record with Djokovic and Federer for the most Grand Slam men's singles titles at the 2022 Australian Open, and became one of four men in history to complete the double Career Grand Slam in singles. As a left-handed player, one of Nadal's main strengths is his forehand, which he hits with a high degree of topspin. He also regularly places among the Tour leaders in percentage of return games, return points, and break points won. Nadal has won the Stefan Edberg Sportsmanship Award five times and was the Laureus World Sportsman of the Year in 2011 and 2021. Time named Nadal one of the 100 most influential people in the world in 2022. He is a recipient of the Grand Cross of Royal Order of Sports Merit, Grand Cross of Order of the Second of May, the Grand Cross of Naval Merit, and the Medal of the City of Paris. Representing Spain, he has won two Olympic gold medals, and led the nation to four Davis Cup titles. Nadal has also opened a tennis academy in Mallorca, and is an active philanthropist.Rafael Nadal Parera was born on 3 June 1986 in Manacor, a town on the island of Mallorca in the Balearic Islands, Spain, to parents Ana María Parera Femenías and Sebastián Nadal Homar. His father is a businessman who owns an insurance company, a glass and window company (Vidres Mallorca), and the famous restaurant Sa Punta. His mother once owned a perfume shop, but gave it up to raise Nadal and his younger sister, María Isabel.[8] One of his uncles, Miguel Ángel Nadal, is a retired professional footballer who played for RCD Mallorca, FC Barcelona and the Spanish national team.[9] As a child, he idolized Barcelona striker Ronaldo, and through his uncle was given access to the Barcelona team dressing room to have a photo taken with the Brazilian.[10] Another of his uncles, tennis coach Toni Nadal, introduced him to that game when he was three years old. Rafael Nadal started to play tennis at the Manacor Tennis Club, where Toni worked as a coach, hitting his first few shots with his uncle.[8][11] Nadal initially found tennis boring compared with football, which he often played on the streets of Manacor with his friends.[8][12] He began to play tennis more consistently when he was five, and Toni quickly realized that his young nephew had both the passion and talent to be a serious player.[11] Nadal usually played tennis in a group, but Toni deliberately picked on him during the sessions, shouting at him rather than the other kids, and making him be the one who picked up the balls and swept the courts afterwards.[8] 
# In his 2011 autobiography, he admitted being afraid of Toni and dreaded having solo practice sessions with him.
# """

# enc = tiktoken.encoding_for_model("gpt-2")
# tokens = enc.encode(text)
# tokens = torch.tensor(tokens)
# tokens = tokens.view(1,-1)

# config = GPTConfig()
# model = GPT(config)
# # # print(model(tokens).shape)
# # generation = model.generate(tokens[:,:100],100)
# # # print(generation.shape)
# # print(enc.decode(generation.squeeze(dim=0).tolist()))

# print(model.configure_optimizer(0,0,[0.9,0.95]))

