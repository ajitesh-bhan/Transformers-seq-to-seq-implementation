import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, dim_model: int, vocab_size: int):
        #dim_model = eg; 512 dimension of embedding vector
        #vocab_size = no. of vocab  words
        super().__init__() 
        self.dim_model = dim_model                      
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, dim_model)
        # nn.Embedding is simple lookup table 

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.dim_model)
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, dim_model: int, seq_len: int, droupout: float ) -> None:
        super().__init__() 
        self.dim_model = dim_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(droupout)

        pe =torch.zeros(seq_len, dim_model)
        # create a matrix of shape(seq_len, dim_model) that is wrt to input sequence length 
        #we use log and exponenial to do poisional encoding part to solve the neumarial computation
        position = torch.arange(0, seq_len, dtype= torch.float).unsqueeze(1)
        #vector os shape 1 
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))
        '''
        PE(pos, 2i) = sin(pos/10000^(2i/dim_model))
        PE(pos, 2i+1) = cos(pos/10000^(2i/dim_model))

        i=0,1,2... dim_model-2
        position = [[0],[1],.. [seq_len-1]]
        div term explained below
        First log term is - (1/dim_model)log(10000) =>  log (1/10000^(1/dim_model))
        FIrst exponential term is e^([0,2,....dim_model-2] (remeber these will be single terms corresponding with i for just for explainiation I haven't writen one )
        result = e^([0,2.. dim_model-2]* log(1/10000^(1/dim_model))
        lets call this term [0,2.. dim_model-2]* log(1/10000^(1/dim_model) as y
        result = e^y
        taking log on both sides
        log(result) = log(e^y)
        log(result)=  y
        log(result) = [0,2.. dim_model-1]* log(1/10000^(1/dim_model)
        lets call this term [0,2.. dim_model-2] as A term and 1/10000^(1/dim_model) as B
        log(result) = Alog (B)
        log(result) = log(B^A)
        result= B^A =  B^([0,2.. dim_model-2]) = (1/10000^(1/dim_model))^B
                    =  1/(10000^(B/dim_model))
        result= 1/(10000^([0,2.. dim_model-2]/dim_model))
        
        it is run for i//2 so every value will be calculated i=0 we get 0 in A term and so on

        PE(pos, 2i) = sin(pos/10000^(2i/dim_model))
        PE(pos, 2i+1) = cos(pos/10000^(2i/dim_model))

        '''
        
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # with unsquezze it will become of dimension 3 (1 * seq_len * dim_model)
        # Batch size * seq_len * dim_model
        self.register_buffer('pe', pe)
        #registers as buffer of the model 

    def forward(self, x):

        #x.shape(1) is nothing but input sequence length so why are we specifying in it
        # becuse there is Max sequence length we get position encoding for max seq length 
        #  but we only add only that wrt to input sequence length rest can be masked there is no need of the position encoding
        x= x+ (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
        #randomy  some embeddings will be hidden for next layer .require_grad is used to tell is that we have no trainable parameter 
        

class Layer_Normalization(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        #eps used because if variance sq beomes too small the outpiut will sjoot ip
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #Mulitplied Learnable parameter for layer normalization
        self.bias = nn.Parameter(torch.zeros(1)) #Added Learnable parameter for layer normalization

    def forward(self, x: torch.Tensor) :
        mean = x.mean(dim= -1, keepdim=True)
        #mean for last dimension becuase there we have the embedding values
        std = x.std(dim= -1, keepdim=True)
        return  self.alpha * (x - mean) / (std + self.eps) + self.bias
    

class Feed_Forward_Block(nn.module):
    def __init__(self, dim_model: int, dim_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1= nn.Linear(dim_model, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2= nn.Linear(dim_ff, dim_model)

    def forward(self,x)
        # (Batch * seq_len * dim_model)  :> (Batch * seq_len * dim_ff) :> (Batch * seq_len * dim_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1)))


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.dim_model =dim_model
        self.num_heads = num_heads
        assert  dim_model % num_heads == 0, "dim_model must be divisible by num_heads"
        self.d_k= dim_model // num_heads

        self.w_q= nn.Linear(dim_model, dim_model) #Wq
        self.w_k = nn.Linear(dim_model, dim_model)#Wk
        self.w_v = nn.Linear(dim_model, dim_model)#Wv
        self.w_o = nn.Linear(dim_model, dim_model) #Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod # "staticmethod we can call without it being instance of class "
    def attention( query, key, value, mask, droupout: nn.Dropout):
        d_k =query.shape[-1] # last dimension of the query   

        attention_scores = (query @ key.transpose(-2,-1))/ math.sqrt(d_k)
        # transpose (-2,-1) will transpose last two dimensions 
        # (Batch * num_head * seq_len* d_k ) @  after tanspose (Batch * num_head * d_k *seq_len* )
        # :> (Batch * num_head * seq_len* seq_len )
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask==0, -1e9)
        attention_scores = attention_scores.soft_mask(dim=-1)
        # (Batch * num*head* seq_length * seq_length)

        if droupout is not None:
            attention_scores = droupout(attention_scores)

        return (attention_scores @ value), attention_scores
        #(Batch * num_head * seq_len* seq_len ) @  (Batch * num_head * seq_len* d_k )
        # :> (Batch * num_head * seq_len* d_k )




    def forward(self, q, k, v, mask=None):
        query= self.w_q(q) # (Batch * seq_len * dim_model) :> (Batch * seq_len * dim_model)
        key= self.w_k(k) # (Batch * seq_len * dim_model) :> (Batch * seq_len * dim_model)
        value = self.w_v(v) # (Batch * seq_len * dim_model) :> (Batch * seq_len * dim_model)

        query =query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1,2)
        #(Batch * seq_len * dim_model):> (Batch * seq_len*  num_heads* d_k) :> (Batch * num_head * seq_len* d_k )
        key= key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1,2)
        value= value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1,2)

        x, self.attention_scores= Multi_Head_Attention.attention(query, key, value, mask, self.dropout)

        x= x.transpose(1,2).contiguous().view(x.shape[0], -1, self.num_heads *  self.d_k)
        # (Batch * num_head * seq_len* d_k ) :> (Batch * seq_len *num_head * d_k ):>(Batch * seq_len *dim_model )

        return self.w_o(x)
        # (Batch * seq_len *dim_model ) :> (Batch * seq_len *dim_model)

    
class Residual_Connection(nn.Module):

    def __init__(self, dropout= float) -> None :
        super().__init__()
        self.dropout = nn.dropout(dropout)
        self.norm=  Layer_Normalization()

    def forward(self, x, sub_layer):
        #sub_layer This is the layer to which the residual connection will be applied.
        # we have input encoding and Multihead attention 
        # x= input encoding and sublayer = Multihead attention 
        #  we firt apply normalization to input encoding and then this is input to 
        # the Multihead attention this is self.sublayer(self.norm(x))  this can also be self.norm(self.sublayer(x))
        #  self.sublayer(self.norm(x)) this is done because  it's common practice to apply normalization before adding the input
        #  now this output after applying droupout is added again to x 
        # hence we have below return function      
        return x + self.dropout(sub_layer(self.norm(x)) )
        
class Encoder_Block(nn.Module):

    def __init__(self, self_attention_block : Multi_Head_Attention, feed_forward_block : Feed_Forward_Block, dropout: float) -> None:
        super().__init__()

        '''
        self_attention_block: An instance of the Multi_Head_Attention class representing the self-attention mechanism.
        feed_forward_block: An instance of the Feed_Forward_Block class representing the feedforward neural network.
        dropout: The dropout probability.
        '''
        self.feed_forward_block = feed_forward_block
        self.self_attention_block = self_attention_block
        self.residual_connections= nn.ModuleList([Residual_Connection(dropout) for _ in range(2)])
        # two residual connections
    def forward(self, x, src_mask):
        x= self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x= self.residual_connections[1](x, self.feed_forward_block)
        return x

# in Paper we we have Nx no. of encoders and decoders hence we have defined Encoder and Decoder bkock and we use Encode and Decode two diffent classes to have Nx no. of them    
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super.__init__()
        self.layers = layers
        self.norm= Layer_Normalization

    def forward(self, x, mask):
        for layer in self.layers:
            x= layer(x, mask)
        return self.norm(x)

        
class Decoder_Block(nn.Module):

    def __init__(self, self_attention_block : Multi_Head_Attention, cross_attention_block : Multi_Head_Attention, feed_forward_block : Feed_Forward_Block, dropout: float) -> None:
        super().__init__()
        self.feed_forward_block = feed_forward_block
        self.self_attention_block = self_attention_block
        self.residual_connections= nn.ModuleList([Residual_Connection(dropout) for _ in range(3)])
        self.cross_attention_block = cross_attention_block

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        #self.self_attention_block(x, x, x, src_mask), it's equivalent to calling self.self_attention_block.forward(x, x, x, src_mask).
        # two masks as we have two different languages 
        x= self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x= self.residual_connections[1](x, lambda x: self.self_attention_block(x, encoder_output, encoder_output, src_mask))
        x= self.residual_connections[2](x, self.feed_forward_block)
    
        return x
    
class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super.__init__()
        self.layers = layers
        self.norm= Layer_Normalization

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x= layer(x,  encoder_output, src_mask, tgt_mask)
        return self.norm(x)


# here vocab size is that of target  translation eg. english to french this is french vocab and we are mapping to french vocab 
class Projection_Layer(nn.Module):

    def __init__(self, dim_model: int, vocab_size: int):
        super.__init__()
        self.proj= nn.Linear(dim_model, vocab_size)


    def forward(self, x):
        # (Batch , seq_length, dim_model) :> (batch, seq_length, vocab_size)
        # this is final layer we want to get output vocab size 
        return torch.log_softmax(self.proj(x), dim= -1)
    
class Treansformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings,
                  tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, 
                  tgt_pos: PositionalEncoding, projection_layer: Projection_Layer):
        super.__init__()
        self.encoder = encoder
        self.decoder= decoder
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.src_embed= src_embed
        self.tgt_embed= tgt_embed
        self.projection_layer = projection_layer

    # we build three methods encode, decode and projrvtion layer becuse we need output of encoder hence we don't define just one method

    def encode(self, src, src_mask):
        src= self.src_embed(src)
        src= self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt= self.tgt_embed(tgt)
        tgt= self.src_pos(tgt)
        return self.decoder(encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)


def buid_Transformer(src_vocab_size: int, tgt_vocab_size: int,
                        src_seq_length: int, tgt_seq_length: int, 
                        dim_model: int= 512, N : int =6, # numberof Encoder and decoder blocks
                        num_heads: int=8, dropout: float= 0.1, 
                        dim_ff: int= 2048): # ffedforward layer expanded dimension 
    
    # Embedding Layers
    src_embed = InputEmbeddings(dim_model, src_vocab_size)
    tgt_embed = InputEmbeddings(dim_model, tgt_vocab_size)

    # Positional Encoding 
    src_pos = PositionalEncoding(dim_model, src_seq_length, dropout)
    tgt_pos = PositionalEncoding(dim_model, tgt_seq_length, dropout)

    #Encoder Block
    encoder_blocks= list
    for _ in range (N):
        encoder_self_attention_block = Multi_Head_Attention(dim_model, num_heads, dropout)
        feed_forward_block = Feed_Forward_Block(dim_model, dim_ff, dropout)
        encoder_block = Encoder_Block(encoder_self_attention_block, feed_forward_block, dropout )
        encoder_blocks.append( encoder_block)


    #Decoder block
    decoder_blocks= list
    for _ in range (N):
        decoder_self_attention_block = Multi_Head_Attention(dim_model, num_heads, dropout)
        decoder_cross_attention_block = Multi_Head_Attention(dim_model, num_heads, dropout)
        feed_forward_block = Feed_Forward_Block(dim_model, dim_ff, dropout)
        decoder_block = Decoder_Block(decoder_self_attention_block,decoder_cross_attention_block, feed_forward_block, dropout )
        decoder_blocks.append( decoder_block)   

    #Create complete decoder and encoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder=  Decoder(nn.ModuleList(decoder_blocks))

    #projection layer mapping to output vocab tft_vocab 
    projection_layer= Projection_Layer(dim_model, tgt_vocab_size)

    #Create final transformer
    transformer = Treansformer(encoder, decoder, src_embed, tgt_embed, src_pos, 
                              tgt_pos, projection_layer)
     
    # Initalize the parameters

    for p in transformer.parameters():
        if p.dim() > 1: nn.init.xavier_uniform(p)
    
    return transformer


    '''
    What is happening in this build_Transformer 
    firt we have src_embed, tgt_embed   
    then I create a list of encoder_list and decoder_lsit becuse I may need multiple encoder and decoders
    in on encoder block:
        we initate intances of  class  Multi_Head_Attention, Feed_Forward_Block 
        these are added as input to Encoder_Block class
        now this appended in lsit, still no x is added or any info 
    we initate the Encoder class using encode function in Transformer class  theat builds this all the encoder blocks
    same wau for decoders then projectuin layer
    
    
    '''
