from torch import nn
from torch.nn import functional as F
import torch

class SeqToSeq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        encoder_output, hidden = self.encoder(src)
        output = self.decoder(encoder_output, hidden, tgt)
        return output

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, device=torch.device('cpu')):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, device=device)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True, device=device)

    def forward(self, src):
        embedded = self.embedding(src)
        output, hidden = self.rnn(embedded)
        return output, hidden
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, token_start_of_sentence=1, max_sentence_len=None, device=torch.device('cpu')):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, device=device)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True, device=device)
        self.linear = nn.Linear(hidden_dim, vocab_size, device=device)
        self.softmax = nn.LogSoftmax(dim=2)
        self.token_start_of_sentence = token_start_of_sentence
        self.max_sentence_len = max_sentence_len

    def forward(self, encoder_outputs, hidden, tgt=None):
        decoder_outputs = []
        batch_size = encoder_outputs.size(0)

        # Start token
        start_tensor = torch.empty(batch_size, 1, dtype=torch.long).fill_(self.token_start_of_sentence)
        decoder_output, hidden = self._forward(start_tensor, hidden)
        decoder_outputs.append(decoder_output)

        if tgt != None: # Force teaching on train
          for i in range(tgt.shape[1] - 1):
            decoder_output, hidden = self._forward(tgt[:, i].unsqueeze(1), hidden)
            decoder_output = self.softmax(decoder_output)
            decoder_outputs.append(decoder_output)
        else:
          for i in range(self.max_sentence_len - 1):
            token_tensor = F.log_softmax(decoder_outputs[-1], dim=-1).topk(1)[1].squeeze(-1) # Transform embedding to index
            decoder_output, hidden = self._forward(token_tensor, hidden)
            decoder_outputs.append(decoder_output)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = self.softmax(decoder_outputs)
        return decoder_outputs
    
    def _forward(self, decoder_output, hidden):
        embedded = self.embedding(decoder_output)
        output, hidden = self.rnn(embedded, hidden)
        output = self.linear(output)
        return output, hidden

