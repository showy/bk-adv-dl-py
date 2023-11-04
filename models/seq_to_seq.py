from torch import nn

class SeqToSeq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        encoder_output, hidden = self.encoder(src)
        output = self.decoder(trg, hidden, encoder_output)
        return output

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, dropout=dropout)

    def forward(self, src):
        embedded = self.embedding(src)
        output, hidden = self.rnn(embedded)
        return output, hidden
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, trg, hidden, encoder_output):
        embedded = self.embedding(trg)
        output, hidden = self.rnn(embedded, hidden)
        output = self.linear(output)
        output = self.softmax(output)
        return output, hidden

