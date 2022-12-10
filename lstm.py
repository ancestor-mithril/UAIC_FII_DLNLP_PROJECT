"""
Adapted from https://github.com/claravania/lstm-pytorch
"""
import torch
import torch.autograd as autograd
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence


class LSTMMultiLabelClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size):
        # TODO: add pretrained embedding weights and freeze them
        super(LSTMMultiLabelClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)

        self.dropout_layer = nn.Dropout(p=0.2)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()  # Sigmoid because it's multi-label


    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
                autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))

    def forward(self, batch, lengths):
        # TODO? self or not?
        hidden = self.init_hidden(batch.size(-1))

        embeds = self.embedding(batch)
        packed_input = pack_padded_sequence(embeds, lengths)
        outputs, (ht, ct) = self.lstm(packed_input, hidden)

        # ht is the last hidden state of the sequences
        # ht = (1 x batch_size x hidden_dim)
        # ht[-1] = (batch_size x hidden_dim)

        output = self.dropout_layer(ht[-1])
        output = self.fc(output)
        output = self.sigmoid(output)
        # ! Loss must be binary cross entropy in this case
        return output
