import torch
import torch.nn as nn
from MultiHeadAttention import AttentionLayer
from MultiHeadAttention import FFNLayer



class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = AttentionLayer(d_model, n_head, dropout)
        self.ffn = FFNLayer(d_model, d_ff, dropout)
        self.cross_attention = AttentionLayer(d_model, n_head, dropout)
        

    def forward(self, x,encoder_output, tgt_mask, src_mask):
        x = self.self_attention(x,x, x, tgt_mask)
        x = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.ffn(x)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model,  d_ff, n_head, n_layers, dropout):
        super(Decoder,self).__init__()
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, d_ff, n_head, dropout) for i in range(n_layers)])

    def forward(self, x, encoder_output, tgt_mask,src_mask):
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        return x
