# iwslt_datamodule.py
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import torch

class IWSLTDataModule:
    def __init__(self, max_len: int, batch_size: int, device: str):
        self.max_len = max_len
        self.batch_size = batch_size
        self.device = device

        self.tokenizer_name = "t5-small"
        self.dataset_name = "iwslt2017"
        self.dataset_config = "iwslt2017-en-it"
        self.seed = 42

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt"
        )

        self.train_loader = None
        self.valid_loader = None

    def setup(self):
        dataset = load_dataset(self.dataset_name, self.dataset_config)

        tokenized_dataset = dataset.map(
            self._preprocess,
            batched=True,
            remove_columns=dataset["train"].column_names
        )

        train_data = tokenized_dataset["train"].shuffle(seed=self.seed)
        valid_data = tokenized_dataset["validation"]

        self.train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collator,
        )

        self.valid_loader = DataLoader(
            valid_data,
            batch_size=self.batch_size,
            collate_fn=self.collator,
        )

    def _preprocess(self, examples):
        src_texts = [ex["en"] for ex in examples["translation"]]
        tgt_texts = [ex["it"] for ex in examples["translation"]]

        return self.tokenizer(
            src_texts,
            text_target=tgt_texts,
            truncation=True,
            max_length=self.max_len,
        )

