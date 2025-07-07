import torch
import torch.nn as nn
from Embeddings import Embeddings
from PositionalEncoding import PositionalEncoding
from Encoder import Encoder 
from Decoder import Decoder


class Transformer(nn.Module):
    def __init__(self,vocab_size, d_model,  d_ff, n_head, n_layers, dropout):
        super(Transformer,self).__init__()
        self.embedding = Embeddings(d_model, vocab_size)
        self.pos_encoder = PositionalEncoding(d_model,dropout)
        self.encoder = Encoder(d_model, d_ff, n_head, n_layers, dropout)
        self.decoder = Decoder(d_model,  d_ff, n_head, n_layers, dropout)
        self.fc = nn.Linear(d_model, vocab_size)
        self._reset_parameters()

    def forward(self,inputs, outputs, src_mask, tgt_mask):
        embedded_inputs = self.embedding(inputs)
        embedded_outputs = self.embedding(outputs)
        embedded_inputs = self.pos_encoder(embedded_inputs)
        embedded_outputs = self.pos_encoder(embedded_outputs)
        encoding = self.encoder(embedded_inputs, src_mask)
        decoding = self.decoder(embedded_outputs, encoding, tgt_mask, src_mask)
        return self.fc(decoding)
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

