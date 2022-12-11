"""
Adapted from https://github.com/claravania/lstm-pytorch
"""
import torch
import torch.autograd as autograd
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence


class LSTMMultiLabelClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, hidden_dim, output_size):
        # TODO: add pretrained embedding weights and freeze them
        super(LSTMMultiLabelClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # TODO: Parametrize this
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=0.2, batch_first=True, bidirectional=True)

        self.num_layers = num_layers

        self.dropout_layer = nn.Dropout(p=0.2)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()  # Sigmoid because it's multi-label

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.randn(4, batch_size, self.hidden_dim)).to(self.fc.weight.device),
                autograd.Variable(torch.randn(4, batch_size, self.hidden_dim)).to(self.fc.weight.device))

    def forward(self, batch):
        self.hidden = self.init_hidden(batch.size(0))
        # self because then it can be zero graded?

        embeds = self.embedding(batch)
        pack_padded_sequence(embeds, torch.tensor([batch.size(1) for _ in range(batch.size(0))]), batch_first=True)
        outputs, (ht, ct) = self.lstm(embeds, self.hidden)

        # ht is the last hidden state of the sequences
        # ht = (1 x batch_size x hidden_dim)
        # ht[-1] = (batch_size x hidden_dim)

        output = self.dropout_layer(ht[-1])
        output = self.fc(output)
        output = self.sigmoid(output)
        # ! Loss must be binary cross entropy in this case
        return output
