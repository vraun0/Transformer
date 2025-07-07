import torch 
from Transformer import Transformer
from transformers import AutoTokenizer 

def inference(model, tokenizer, input_text, max_len=128, device='cpu'):
    model.to(device)
    model.eval()
    tokenized_input = tokenizer(input_text, return_tensors="pt")
    src = tokenized_input['input_ids'].to(device)

    src_mask = (src == tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2).to(device)
    
    output_tokens = torch.tensor([[tokenizer.pad_token_id]], device=device).long()

    with torch.no_grad():
        embedded_src = model.pos_encoder(model.embedding(src))
        memory = model.encoder(embedded_src, src_mask)

        for i in range(max_len - 1):
            tgt_len = output_tokens.size(1)
            tgt_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=device), diagonal=1).bool()

            embedded_tgt = model.pos_encoder(model.embedding(output_tokens))
            decoding_output = model.decoder(embedded_tgt, memory, tgt_mask, src_mask)

            logits = model.fc(decoding_output)
            last_token_logits = logits[:, -1, :] 

            pred_token = torch.argmax(last_token_logits, dim=-1).item() 
            
            
            if pred_token == tokenizer.eos_token_id:
                break
            output_tokens = torch.cat(
                (output_tokens, torch.tensor([[pred_token]], device=device).long()), dim=1
            )
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)

def preprocess(examples):
    src_texts = [ex['en'] for ex in examples['translation']]
    tgt_texts = [ex['it'] for ex in examples['translation']]
    
    model_inputs = tokenizer(
        src_texts,
        text_target=tgt_texts,
        truncation=True,
        max_length=MAX_LEN
    )
    return model_inputs



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 256
D_MODEL = 256
N_HEAD = 8
N_LAYERS = 4
D_FF = 1024
DROPOUT = 0.1
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
EPOCHS = 70
MAX_LEN = 128
PATIENCE = 10 
SAVE_PATH = "models/transformer_en_it_2.pt"


tokenizer = AutoTokenizer.from_pretrained("t5-small")
VOCAB_SIZE = tokenizer.vocab_size



inference_model = Transformer(VOCAB_SIZE, D_MODEL, D_FF, N_HEAD, N_LAYERS, DROPOUT)
inference_model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
english = input("Enter English text: ")
italian = inference(inference_model, tokenizer, english,device = DEVICE)
print(italian)
