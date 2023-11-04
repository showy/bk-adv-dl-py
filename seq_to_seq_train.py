from utils.basic_tokenizer import Tokenizer
from models.seq_to_seq import SeqToSeq, Encoder, Decoder
from datasets import load_dataset
from torch import nn

ds_train = load_dataset('opus100', 'en-fr', split='train[:10%]')
ds_test = load_dataset('opus100', 'en-fr', split='test')

eng_tokenizer = Tokenizer()
fr_tokenizer = Tokenizer()

samples = 100000

ds_train_ids = ds_train.map(lambda e: dict(
  eng_id = eng_tokenizer.tokenize(e['translation']['en']),
  fr_id = fr_tokenizer.tokenize(e['translation']['fr'])
), batched=False)

eng_tokenizer.freeze_vocab()
fr_tokenizer.freeze_vocab()

ds_test_ids = ds_test.map(lambda e: dict(
  eng_id = eng_tokenizer.tokenize(e['translation']['en']),
  fr_id = fr_tokenizer.tokenize(e['translation']['fr'])
), batched=False)

ENG_VOCAB_SIZE = len(eng_tokenizer)
FR_VOCAB_SIZE = len(fr_tokenizer)
EMBEDDING_DIM = 128

encoder = Encoder(ENG_VOCAB_SIZE, EMBEDDING_DIM, 256, 2, 0.5)
decoder = Decoder(FR_VOCAB_SIZE, EMBEDDING_DIM, 256, 2, 0.5)

model = SeqToSeq(encoder, decoder)
