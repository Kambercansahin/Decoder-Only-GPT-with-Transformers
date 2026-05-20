import torch

from torch import nn
import torch.nn.functional as F

device = "mps" if torch.backends.mps.is_available() else "cpu"

class Decoder_block(nn.Module):
    def __init__(self,embedding_dim:int=384,number_heads:int=6,drop_outs:float=0.1):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(normalized_shape=embedding_dim)#LN

        #for masked msa
        self.msa_mask = nn.MultiheadAttention(embed_dim=embedding_dim,num_heads=number_heads,dropout=drop_outs,batch_first=True)

        self.layer_norm2 = nn.LayerNorm(normalized_shape=embedding_dim)

        #Mlp block in paper -->linear,Relu(Gelu)-->linear -->dropout
        self.mlp_block = nn.Sequential(
            nn.Linear(in_features=embedding_dim,out_features= 4* embedding_dim),
            nn.GELU(),
            nn.Linear(in_features=4*embedding_dim,out_features=embedding_dim),
            nn.Dropout(drop_outs)
        )

    def forward(self,x:torch.Tensor,causal_mask:torch.Tensor):
        #pass the input through the Layer Norm layer first.
        x_norm = self.layer_norm1(x)

        #take the  attn_output ,and attn_output_weights(_), I use my causal_mask so is_causal=False
        attn_output,_ = self.msa_mask(query=x_norm,value=x_norm,key=x_norm,attn_mask=causal_mask,is_causal=False)

        x = x + attn_output #resudal + msa

        x = x + self.mlp_block(self.layer_norm2(x))

        return x


class GPT_Transformers_Block(nn.Module):
    def __init__(self,vocab_size:int=65,embedding_dim:int=384,number_heads:int=6,num_layers:int=6,drop_outs:float=0.1,block_size:int=256):
        super().__init__()

        self.block_size = block_size
        # for each characters create embedding, [1x384] -> Output shape: [batch_size, block_size, embedding_dim]
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)#[5x256x384]

        #For all characters  each characters [1x384] each block_size 256 so positional encoding shape =[256x384]
        self.positional_embedding = nn.Embedding(num_embeddings=block_size,embedding_dim=embedding_dim)#[5x256x384]

        self.decoder_block = nn.ModuleList([
            Decoder_block(embedding_dim=embedding_dim,number_heads=number_heads,
                          drop_outs=drop_outs)
            for _ in range(num_layers) ])#in paper num_of layers 12 but I use 6 with nn.ModuleList

        self.final_LN = nn.LayerNorm(normalized_shape=embedding_dim)#in attention all you needs paper

        #I have 65 unique characters but my embedding_size =384 so I convert to 65 with linear
        self.linear_last_for_output = nn.Linear(in_features=embedding_dim,out_features=vocab_size) #[5x256x65]

        causal_mask = torch.triu(torch.ones(block_size,block_size,dtype=torch.bool),diagonal=1)#causal mask [256x256] True false

        self.register_buffer("causal_mask",causal_mask)

        self.loss_fn = nn.CrossEntropyLoss() #Calculate loss_fn

        self.drop_out=nn.Dropout(drop_outs)#in papers we use drop out added positional and token embeddings
    def forward(self,x:torch.Tensor,y:torch.Tensor=None):#caluculate loss fn so I needs y(target)
        batch_size,seq_len = x.shape

        token_em = self.token_embedding(x)#[5x256x384]

        position= torch.arange(seq_len,device=device) #256 index number for each char for in block_size

        position_emd = self.positional_embedding(position)

        x = self.drop_out(token_em+position_emd)#text and positional embedding

        mask = self.causal_mask[:seq_len,:seq_len] # I choose seq_len instead of block_size sometimes in contex last size smaller than block size

        for blocks in self.decoder_block: #I have 6 times blocks
            x = blocks(x,mask)

        x = self.final_LN(x)

        x = self.linear_last_for_output(x)

        #Now I calculate loss_fn for y

        loss= None
        if y is not None: #loss fn for x [NxC],for y [N] N-->batch_size,C-->number of class
            batch_size,seq_len,vocab_size = x.shape

            x_flats = x.reshape(batch_size*seq_len,vocab_size)#[5*256,65]
            y_flat = y.reshape(batch_size*seq_len)#[5*256]

            loss = self.loss_fn(x_flats,y_flat)

        return x, loss

    @torch.no_grad() #non grad function
    def generation(self,x:torch.Tensor,max_new_token:int,temperature:float=0.5):
        #max_new_token = How many tokens will GPT
        #temperature is hallucination of models if greater than 1 hallucination best 0-1

        for _ in range(max_new_token):
            if x.size(1) < self.block_size: #if user prompt < 256 token
                current_input = x
            else:
                current_input = x[:, -self.block_size] #if user prompt >256 token choose last 256 token

            x_logits , loss = self.forward(current_input)#we use only x so we don' t need loss in this case because that is generation

            x_logits_last = x_logits[:,-1,:]#we choose last char for predict , because model predict next char
            x_logits_last = x_logits_last / temperature

            probs = F.softmax(x_logits_last,dim=-1) # logits with softmax

            next_token = torch.multinomial(probs,num_samples=1)#we use multinomial because if we choose always high rate ,model always create same char

            x = torch.cat([x,next_token],dim=1) # added next token in my input

        return x
