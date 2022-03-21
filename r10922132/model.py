from pickle import NONE
from typing import Dict, List

import torch
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        rnn_type = "GRU"
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.rnn_type = rnn_type

        if rnn_type == "GRU":
            self.rnn = torch.nn.GRU(
                input_size = 300,
                hidden_size = hidden_size,
                num_layers = num_layers,
                batch_first = True,
                dropout = dropout,
                bidirectional = bidirectional            
            )
        else:
            self.rnn = torch.nn.LSTM(
                input_size = 300,
                hidden_size = hidden_size,
                num_layers = num_layers,
                batch_first = True,
                dropout = dropout,
                bidirectional = bidirectional            
            )

        self.fc = torch.nn.Linear(self.encoder_output_size, num_class)    

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        #raise NotImplementedError
        if self.bidirectional == True:
            return self.hidden_size * 2
        else:
            return self.hidden_size

    def forward(self, batch) -> torch.Tensor:
        # TODO: implement model forward
        #raise NotImplementedError
        batch = self.embed(batch)
        output, _ = self.rnn(batch, None)
        output = self.fc(output[:,-1,:])
           
        return output

class SeqSlot(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        rnn_type = "GRU"
    ) -> None:
        super(SeqSlot, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.rnn_type = rnn_type

        if rnn_type == "GRU":
            self.rnn = torch.nn.GRU(
                input_size = 300,
                hidden_size = hidden_size,
                num_layers = num_layers,
                batch_first = True,
                dropout = dropout,
                bidirectional = bidirectional            
            )
        else:
            self.rnn = torch.nn.LSTM(
                input_size = 300,
                hidden_size = hidden_size,
                num_layers = num_layers,
                batch_first = True,
                dropout = dropout,
                bidirectional = bidirectional            
            )

        self.projection = torch.nn.Linear(self.encoder_output_size, num_class)   

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        #raise NotImplementedError
        if self.bidirectional == True:
            return self.hidden_size * 2
        else:
            return self.hidden_size

    def forward(self, batch) -> torch.Tensor:
        # TODO: implement model forward
        #raise NotImplementedError
        batch = self.embed(batch)
        output, _ = self.rnn(batch, None)
        output = output.reshape(-1, output.shape[2])
        output = self.projection(output)
           
        return output
