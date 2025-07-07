import torch 
import torch.nn as nn
from MultiHeadAttention import AttentionLayer
from MultiHeadAttention import FFNLayer

class EncoderLayer(nn.Module):
    def __init__(self, d_model,  d_ff, n_head, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = AttentionLayer(d_model, n_head, dropout)
        self.ffn = FFNLayer(d_model, d_ff, dropout)
        

    def forward(self, x, src_mask):
        x = self.self_attention(x,x,x, src_mask)
        x = self.ffn(x)
        return x
        

class Encoder(nn.Module):
    def __init__(self, d_model,  d_ff, n_head, n_layers, dropout):
        super(Encoder,self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model,  d_ff,n_head, dropout) for _ in range(n_layers)])

    def forward(self, src, src_mask): 
        x = src
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x       
        
        
