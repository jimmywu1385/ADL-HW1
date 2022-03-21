from ast import arg
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from sklearn.utils import shuffle

import torch

from dataset import SeqSlotDataset
from model import SeqSlot
from utils import Vocab

import csv
def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqSlotDataset(data, vocab, tag2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    
    model = SeqSlot(
                    embeddings, args.hidden_size, args.num_layers,
                    args.dropout, args.bidirectional, len(tag2idx), args.rnn_type
            ).to(args.device)

    # load weights into model
    mckpt = torch.load(args.ckpt_path)
    model.load_state_dict(mckpt)
    model.eval()

    # TODO: predict dataset
    pred_list = []
    id_list = []
    for batch in dataloader:
        tokens = batch["tokens"].to(args.device) 
        ids = batch["id"]
        seq_lens = batch["seq_len"]

        with torch.no_grad():
            pred = model(tokens)            
            predicted = pred.argmax(dim=-1).view(len(tokens), -1) 
            for pred, seq_len, id in zip(predicted, seq_lens, ids):
                pred = pred[:seq_len]
                pred_list.append(" ".join([dataset.idx2label(i.item()) for i in pred]))
                id_list.append(id)
    
    # COMPLETE

    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        
        writer.writerow(['id', 'tags'])
        for i in range(len(id_list)):
            writer.writerow([id_list[i], pred_list[i]])


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

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
    main(args)
