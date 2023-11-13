class Tokenizer():
    def __init__(self, max_seq_len = 32):
        import re
        self.start_tkn = '[START]'
        self.end_tkn = '[END]'
        self.unk_tkn = 'UNK'
        self.pad_tkn = 'PAD'
        self.tkn_to_id = {self.start_tkn: 3, self.end_tkn: 1, self.unk_tkn: 2, self.pad_tkn: 0}
        self.id_to_tkn = {v: k for k, v in self.tkn_to_id.items()}
        self.split_regex = re.compile(r'\W+')
        self.next_id = 4
        self.is_vocab_frozen = False
        self.vocab_max_size = 100000
        self.max_seq_len = max_seq_len

    def get_start_token(self):
        return self.start_tkn
    
    def get_start_token_id(self):
        return self.tkn_to_id[self.start_tkn]
    
    def __len__(self):
        return max(self.id_to_tkn.keys()) + 1 # First index is 0
    
    def split(self, sentence):
        return self.split_regex.split(sentence)

    def freeze_vocab(self):
        self.is_vocab_frozen = True

    def add_token(self, token):
        if token not in self.tkn_to_id:
            self.tkn_to_id[token] = self.next_id
            self.id_to_tkn[self.next_id] = token
            self.next_id += 1

        return self.tkn_to_id[token]

    def get_id(self, token):
        if token in self.tkn_to_id:
            return self.tkn_to_id[token]
        else:
            if self.is_vocab_frozen:
                return self.tkn_to_id[self.unk_tkn]
            else:
                return self.add_token(token)

    def get_tkn(self, id):
        if id in self.id_to_tkn:
            return self.id_to_tkn[id]
        else:
            return self.unk_tkn

    def untokenize(self, ids):
        return [self.get_tkn(_id) for _id in ids]

    def tokenize(self, sentence, pad=True):
        sentence = self.split(sentence)
#        sentence_tknized = [self.tkn_to_id[self.start_tkn]] + [self.get_id(tkn) for tkn in sentence] + [self.tkn_to_id[self.end_tkn]]
        sentence_tknized = [self.get_id(tkn) for tkn in sentence] + [self.tkn_to_id[self.end_tkn]] # Not adding start token

        pad_tkns = self.max_seq_len - len(sentence_tknized)
        if pad_tkns > 0 and pad:
            sentence_tknized += [self.tkn_to_id[self.pad_tkn]] * pad_tkns
        return sentence_tknized[:self.max_seq_len]