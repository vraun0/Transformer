import torch.nn as nn
import torch
from tqdm.auto import tqdm

def create_masks(src_batch, tgt_batch, pad_token_id, device):
    src_mask = (src_batch == pad_token_id).unsqueeze(1).unsqueeze(2)
    tgt_pad_mask = (tgt_batch == pad_token_id).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt_batch.size(1)
    tgt_causal_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=device), diagonal=1).bool()
    tgt_mask = tgt_pad_mask | tgt_causal_mask
    return src_mask.to(device), tgt_mask.to(device)



def train_epoch(model, dataloader, optimizer, criterion, pad_token_id, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        src = batch['input_ids']
        labels = batch['labels']
        
        tgt = labels.clone()
        tgt[tgt == -100] = pad_token_id 
        shifted_tgt = torch.full_like(tgt, pad_token_id)
        shifted_tgt[:, 1:] = tgt[:, :-1]
        tgt = shifted_tgt
        
        src_mask, tgt_mask = create_masks(src, tgt, pad_token_id, device)
        
        optimizer.zero_grad()
        output = model(src, tgt, src_mask, tgt_mask)
        
        output_flat = output.view(-1, output.shape[-1])
        labels_flat = labels.view(-1)
        loss = criterion(output_flat, labels_flat)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, pad_token_id, device):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc='Evaluating', leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            src = batch['input_ids']
            labels = batch['labels']
            
            tgt = labels.clone()
            tgt[tgt == -100] = pad_token_id 
            shifted_tgt = torch.full_like(tgt, pad_token_id)
            shifted_tgt[:, 1:] = tgt[:, :-1]
            tgt = shifted_tgt
            
            src_mask, tgt_mask = create_masks(src, tgt, pad_token_id, device)
            
            output = model(src, tgt, src_mask, tgt_mask)
            output_flat = output.view(-1, output.shape[-1])
            labels_flat = labels.view(-1)
            loss = criterion(output_flat, labels_flat)
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
    return total_loss / len(dataloader)    
