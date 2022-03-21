import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
import string
from typing import Any, Dict
from unittest import TestLoader

import torch
from tqdm import trange

from dataset import SeqClsDataset, SeqSlotDataset
from utils import Vocab

from model import SeqClassifier, SeqSlot

import csv
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
#max token in data = 35
TRAIN = "train"
DEV = "eval"
TEST = "test"
SPLITS = [TRAIN, DEV]

def main(args):
    seed = 126
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqSlotDataset] = {
        split: SeqSlotDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    train_datasets = torch.utils.data.DataLoader(datasets["train"], batch_size=args.batch_size, collate_fn=datasets["train"].collate_fn, shuffle=True)
    eval_datasets = torch.utils.data.DataLoader(datasets["eval"], batch_size=args.batch_size, collate_fn=datasets["eval"].collate_fn, shuffle=False)

    embeddings = torch.load(str(args.cache_dir) + "/embeddings.pt")

    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqSlot(
                    embeddings, args.hidden_size, args.num_layers,
                    args.dropout, args.bidirectional, len(tag2idx), args.rnn_type
            ).to(args.device)

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    epoch_pbar = trange(args.num_epoch, desc="Epoch")

    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        train_loss = 0.0
        correct =0.0
        total =0.0
        for i, dic in enumerate(train_datasets):
            tokens = dic["tokens"].to(args.device)
            tags = dic["tags"].to(args.device)
            seq_lens = dic["seq_len"]

            optimizer.zero_grad()

            pred = model(tokens)
            loss = criterion(pred, tags.view(-1))
            loss.backward()
            optimizer.step()

            predicted = pred.argmax(dim=-1).view(len(tokens), -1)
            tags = tags.view(len(tokens), -1)
            for pred, tag, seq_len in zip(predicted, tags, seq_lens):                
                pred, tag = pred[:seq_len], tag[:seq_len]
                corrects = pred.eq(tag)
                correct += (torch.all(corrects)).int()
                total += 1           

            train_loss += loss.item()
            if i % 10 == 0:
                print(f"epoch : {epoch + 1}, iter : {i + 1:5d} loss: {train_loss / (i+1):.5f}")

        print(f"Accuracy : {(correct/total)}")
        # TODO: Evaluation loop - calculate accuracy and save model weights
        print("------eval start------\n")
        
        model.eval()
        eval_loss = 0.0
        correct =0.0
        total =0.0
        y_pred = []
        y_true = []
        for i, dic in enumerate(eval_datasets):
            tokens = dic["tokens"].to(args.device)
            tags = dic["tags"].to(args.device)
            seq_lens = dic["seq_len"]

            with torch.no_grad():
                pred = model(tokens)
                loss = criterion(pred, tags.view(-1))
            
                predicted = pred.argmax(dim=-1).view(len(tokens), -1)
                tags = tags.view(len(tokens), -1)
                for pred, tag, seq_len in zip(predicted, tags, seq_lens):
                    pred, tag = pred[:seq_len], tag[:seq_len]
                    corrects = pred.eq(tag)
                    correct += (torch.all(corrects)).int()
                    total += 1
                    y_pred.append([datasets["eval"].idx2label(i.item()) for i in pred])
                    y_true.append([datasets["eval"].idx2label(i.item()) for i in tag])
            eval_loss += loss.item()
            if i % 10 == 0:
                print(f"epoch : {epoch + 1}, iter : {i + 1:5d} loss: {eval_loss / (i+1):.5f}")
        print(f"Accuracy : {correct/total}")
        report = classification_report(y_pred, y_true, mode='strict', scheme=IOB2)
        print(report)
    
    print("DONE\n")

    # TODO: Inference on test set  
    torch.save(model.state_dict(), args.ckpt_dir / (args.model_name+".pt"))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=None)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--rnn_type", type=str, default="GRU")

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    # save model
    parser.add_argument("--model_name", type=str, default="best")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
