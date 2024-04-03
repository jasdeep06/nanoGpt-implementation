from dataclasses import dataclass
import torch.nn as nn
import tiktoken
import torch
import torch.nn.functional as F




@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    num_transformer_block: int = 12
    num_transformer_heads: int = 12
    n_embed: int = 768
    dropout: int = 0
    bias: bool = True


class LayerNorm(nn.Module):
    def __init__(self,ndim,bias):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self,x):
        return F.layer_norm(x,self.weights.shape,self.weights,self.bias,1e-5)
    

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.layer_norm = LayerNorm(config.n_embed,config.bias)
        self.attention = CausalAttention(config)
        self.layer_norm = LayerNorm(config.n_embed,config.bias)
        self.mlp = MLP(config)
    
    def forward(self,x):
        x = x + self.attention(self.layer_norm(x))
        x = x + self.mlp(self.layer_norm(x))
        return x
    
class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c1 = nn.Linear(config.n_embed,4*config.n_embed)
        self.act1 = nn.GELU()
        self.c2 = nn.Linear(4*config.n_embed,config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x):
        x = self.c1(x)
        x = self.act1(x)
        x = self.c2(x)
        x = self.dropout(x)

        return x


class CausalAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.input_projection = nn.Linear(config.n_embed,3*config.n_embed)
        self.register_buffer('bias',torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))
        self.output_projection = nn.Linear(config.n_embed,config.n_embed)

    def forward(self,x):
        b,t,c = x.size()
        x = self.input_projection(x)
        k,q,v = torch.split(x,config.n_embed,dim=2)
        # print(k.shape,q.shape,v.shape)
        k = k.view(b,t,config.num_transformer_heads,c//config.num_transformer_heads).transpose(2,1)
        q = q.view(b,t,config.num_transformer_heads,c//config.num_transformer_heads).transpose(2,1)
        v = v.view(b,t,config.num_transformer_heads,c//config.num_transformer_heads).transpose(2,1)
        # print(k.shape,q.shape,v.shape)

        attn_scores = q @ k.transpose(-2,-1)
        attn_scores = attn_scores.masked_fill(self.bias[:,:,:t,:t] == 0,float('-inf'))
        attn_scores = F.softmax(attn_scores,dim=-1)
        # print(attn_scores.shape)
        y = attn_scores @ v
        # print(y.shape)
        y = y.transpose(1,2).contiguous().view(b,t,c)

        y = self.output_projection(y)

        # print(y.shape)
        return y



config = GPTConfig()

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size,config.n_embed)
        self.pos_embedding = nn.Embedding(config.block_size,config.n_embed)
        self.trans_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_transformer_block)])
        self.layer_norm = LayerNorm(config.n_embed,config.bias)
        self.lm_head = nn.Linear(config.n_embed,config.vocab_size)

        #weight tying
        self.token_embedding.weight = self.lm_head.weight

        #initializing weights
        self.apply(self._init_weights)



    def forward(self,idx,targets=None):
        b,t = idx.size()
        x = self.token_embedding(idx)
        reference = torch.arange(0,t,dtype=torch.long)
        pos = self.pos_embedding(reference)
        x = self.trans_dropout(x + pos)
        for block in self.blocks:
            x = block(x)
        
        x = self.layer_norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None


        return logits,loss
    
    
    
    def generate(self,idx,max_new_tokens,temperature=1.0,top_k=None):

        for _ in range(max_new_tokens):
            output = self(idx)
            relevant_output = output[:,-1,:]/temperature
            probs = F.softmax(relevant_output,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1) 
        return idx
    
    def get_num_parameters(self,non_embedding=True):
        total_parameters = 0
        for parameter in self.parameters():
            total_parameters += parameter.data.numel()
        if non_embedding:
            total_parameters -= self.pos_embedding.weight.numel()
            #As we are using weight tying, the token embedding weights will be used as weights in the final layer
            #So we don't exclude them
            # total_parameters -= self.token_embedding.weight.numel()

        return total_parameters

    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)

    
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
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for pn, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

        return optimizer
    


        












text = """
Rafael Nadal Parera a Spanish professional tennis player. He has been ranked world No. 1 in singles by the Association of Tennis Professionals (ATP) for 209 weeks, and has finished as the year-end No. 1 five times. Nadal has won 22 Grand Slam men's singles titles, including a record 14 French Open titles. He has won 92 ATP-level singles titles, including 36 Masters titles and an Olympic gold medal, with 63 of these on clay courts. Nadal is one of only two men to complete the Career Golden Slam in singles.[b] His 81 consecutive wins on clay constitute the longest single-surface win streak in the Open Era. For over a decade, Nadal has led men's tennis along with Roger Federer and Novak Djokovic as the Big Three.[c] At the start of his professional career, Nadal became one of the most successful teenagers in ATP Tour history, reaching the world No. 2 ranking and winning 16 titles before turning 20, including his first French Open and six Masters events. Nadal became the world No. 1 for the first time in 2008 after defeating Federer in a historic Wimbledon final, his first major victory off clay. He followed up his win with an Olympic singles gold at the 2008 Beijing Olympics. After defeating Djokovic in the 2010 US Open final, then-24-year-old Nadal became the youngest man in the Open Era to achieve the Career Grand Slam, and the first man to win majors on three different surfaces (hard, grass, and clay) in the same year (Surface Slam).After two injury-plagued seasons, Nadal returned to the Tour in 2013, reaching 14 finals, winning two majors and five Masters events including the US Open Series sweep (Summer Slam). He continued his dominance at the French Open, securing six titles, two US Open titles, an Australian Open title, and an Olympic doubles gold at the 2016 Rio Olympics with Marc López. Nadal surpassed his joint-record with Djokovic and Federer for the most Grand Slam men's singles titles at the 2022 Australian Open, and became one of four men in history to complete the double Career Grand Slam in singles. As a left-handed player, one of Nadal's main strengths is his forehand, which he hits with a high degree of topspin. He also regularly places among the Tour leaders in percentage of return games, return points, and break points won. Nadal has won the Stefan Edberg Sportsmanship Award five times and was the Laureus World Sportsman of the Year in 2011 and 2021. Time named Nadal one of the 100 most influential people in the world in 2022. He is a recipient of the Grand Cross of Royal Order of Sports Merit, Grand Cross of Order of the Second of May, the Grand Cross of Naval Merit, and the Medal of the City of Paris. Representing Spain, he has won two Olympic gold medals, and led the nation to four Davis Cup titles. Nadal has also opened a tennis academy in Mallorca, and is an active philanthropist.Rafael Nadal Parera was born on 3 June 1986 in Manacor, a town on the island of Mallorca in the Balearic Islands, Spain, to parents Ana María Parera Femenías and Sebastián Nadal Homar. His father is a businessman who owns an insurance company, a glass and window company (Vidres Mallorca), and the famous restaurant Sa Punta. His mother once owned a perfume shop, but gave it up to raise Nadal and his younger sister, María Isabel.[8] One of his uncles, Miguel Ángel Nadal, is a retired professional footballer who played for RCD Mallorca, FC Barcelona and the Spanish national team.[9] As a child, he idolized Barcelona striker Ronaldo, and through his uncle was given access to the Barcelona team dressing room to have a photo taken with the Brazilian.[10] Another of his uncles, tennis coach Toni Nadal, introduced him to that game when he was three years old. Rafael Nadal started to play tennis at the Manacor Tennis Club, where Toni worked as a coach, hitting his first few shots with his uncle.[8][11] Nadal initially found tennis boring compared with football, which he often played on the streets of Manacor with his friends.[8][12] He began to play tennis more consistently when he was five, and Toni quickly realized that his young nephew had both the passion and talent to be a serious player.[11] Nadal usually played tennis in a group, but Toni deliberately picked on him during the sessions, shouting at him rather than the other kids, and making him be the one who picked up the balls and swept the courts afterwards.[8] 
In his 2011 autobiography, he admitted being afraid of Toni and dreaded having solo practice sessions with him.
"""

enc = tiktoken.encoding_for_model("gpt-2")
tokens = enc.encode(text)
tokens = torch.tensor(tokens)
tokens = tokens.view(1,-1)

config = GPTConfig()
model = GPT(config)
# # print(model(tokens).shape)
# generation = model.generate(tokens[:,:100],100)
# # print(generation.shape)
# print(enc.decode(generation.squeeze(dim=0).tolist()))

print(model.configure_optimizer(0,0,[0.9,0.95]))

