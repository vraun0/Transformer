import torch 
import torch.nn as nn 
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_model // n_head
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask):
        batch_size = Q.size(0)

        q = self.q_proj(Q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.k_proj(K).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_proj(V).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        p_attn = scores.softmax(dim=-1)
        attention_output = torch.matmul(p_attn, v)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_k)
        return self.out_proj(attention_output)

        
    

class AttentionLayer(nn.Module):
    def __init__(self,d_model, n_head, dropout):
        super(AttentionLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, n_head)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q,k,v, mask):
        out = self.mha(q,k,v, mask)
        out = self.dropout(out)
        out = out+q
        out = self.layer_norm(out)
        return out

class FFNLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(FFNLayer,self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out +x 
        out = self.layer_norm(out)
        return out
