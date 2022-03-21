from typing import List, Dict
from torch import tensor, LongTensor

from torch.utils.data import Dataset

from utils import Vocab


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.batch = dict()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        #raise NotImplementedError
        self.batch["text"] = LongTensor(self.vocab.encode_batch([[word for word in i["text"].split()] for i in samples], self.max_len))
        self.batch["id"] = [i["id"] for i in samples]
        if self.batch["id"][0][:4] != "test":
            self.batch["intent"] = LongTensor([self.label2idx(dic["intent"]) for dic in samples])
        return self.batch

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

class SeqSlotDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.batch = dict()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        #raise NotImplementedError
        self.batch["seq_len"] = [len(i["tokens"]) for i in samples]
        self.batch["tokens"] = LongTensor(self.vocab.encode_batch([i["tokens"] for i in samples], self.max_len))
        self.batch["id"] = [i["id"] for i in samples]
        if self.batch["id"][0][:4] != "test":
            self.batch["tags"] = LongTensor(self.encode_batch([i["tags"] for i in samples], self.max_len))
        return self.batch

    def encode_batch(
        self, batch_tokens: List[List[str]], to_len: int = None
    ) -> List[List[int]]:
        batch_ids = [[self.label2idx(token) for token in tokens] for tokens in batch_tokens]
        to_len = max(len(ids) for ids in batch_ids) if to_len is None else to_len
        output = [seq[:to_len] + [-1] * max(0, to_len - len(seq)) for seq in batch_ids]
        return output

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
