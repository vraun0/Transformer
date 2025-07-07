import torch
from IWSLT_datamodule import IWSLTDataModule
import config  
from Transformer import Transformer
from tqdm.auto import tqdm
import time
from helperFunctions import train_epoch, evaluate 
import torch.nn as nn


DEVICE = config.DEVICE if torch.cuda.is_available() else "cpu"
D_MODEL = config.D_MODEL
N_HEAD = config.N_HEAD
N_LAYERS = config.N_LAYERS
D_FF = config.D_FF
DROPOUT = config.DROPOUT
BATCH_SIZE = config.BATCH_SIZE
LEARNING_RATE = config.LEARNING_RATE
EPOCHS = config.EPOCHS
MAX_LEN = config.MAX_LEN
PATIENCE = config.PATIENCE
SAVE_PATH = config.SAVE_PATH

dm = IWSLTDataModule(MAX_LEN, BATCH_SIZE, DEVICE)
dm.setup()

train_loader = dm.train_loader
valid_loader = dm.valid_loader
VOCAB_SIZE = IWSLTDataModule.tokenizer.vocab_size
PAD_TOKEN_ID = IWSLTDataModule.tokenizer.pad_token_id

model = Transformer(VOCAB_SIZE, D_MODEL, D_FF, N_HEAD, N_LAYERS, DROPOUT).to(DEVICE)
    
criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,      
    'min',          
    factor=0.5,     
    patience=5,     
)

print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

print("Starting training...")
best_valid_loss = float('inf')
epochs_no_improve = 0

for epoch in tqdm(range(1, EPOCHS + 1), desc="Epochs"):
    start_time = time.time()
    
    train_loss = train_epoch(model, train_loader, optimizer, criterion, PAD_TOKEN_ID, DEVICE)
    valid_loss = evaluate(model, valid_loader, criterion, PAD_TOKEN_ID, DEVICE)

    scheduler.step(valid_loss) 
    
    end_time = time.time()
    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
    
    print(f"--- Epoch {epoch}/{EPOCHS} | Time: {int(epoch_mins)}m {int(epoch_secs)}s ---")
    print(f"\tTrain Loss: {train_loss:.4f}")
    print(f"\tValid Loss: {valid_loss:.4f}")

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), SAVE_PATH)
        print("\t-> Saved best model (based on validation loss)")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"\t-> No improvement in validation loss for {epochs_no_improve} epoch(s).")

    if epochs_no_improve >= PATIENCE:
        print(f"\nEarly stopping triggered after {PATIENCE} epochs with no improvement.")
        break

print("Training finished.")

