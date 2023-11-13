from utils.basic_tokenizer import Tokenizer
from models.seq_to_seq import SeqToSeq, Encoder, Decoder
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from torchinfo import summary

SENTENCE_MAX_LEN = 16
TORCH_DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ds_train = load_dataset('opus100', 'en-fr', split='train[:10%]')
ds_test = load_dataset('opus100', 'en-fr', split='test')

eng_tokenizer = Tokenizer(max_seq_len = SENTENCE_MAX_LEN)
fr_tokenizer = Tokenizer(max_seq_len = SENTENCE_MAX_LEN)

samples = 100000

ds_train = ds_train.filter(lambda e: len(e['translation']['en']) < SENTENCE_MAX_LEN and len(e['translation']['fr']) < SENTENCE_MAX_LEN)
ds_test = ds_test.filter(lambda e: len(e['translation']['en']) < SENTENCE_MAX_LEN and len(e['translation']['fr']) < SENTENCE_MAX_LEN)

ds_train_ids = ds_train.map(lambda e: dict(
  eng_id = eng_tokenizer.tokenize(e['translation']['en']),
  fr_id = fr_tokenizer.tokenize(e['translation']['fr']),
), batched=False, load_from_cache_file=False)

eng_tokenizer.freeze_vocab()
fr_tokenizer.freeze_vocab()

ds_test_ids = ds_test.map(lambda e: dict(
  eng_id = eng_tokenizer.tokenize(e['translation']['en']),
  fr_id = fr_tokenizer.tokenize(e['translation']['fr'])
), batched=False, load_from_cache_file=False)

ds_train_ids.set_transform(lambda _batch: {
  k: torch.tensor(v, dtype=torch.long, device=TORCH_DEV) for k, v in _batch.items()
}, columns=['eng_id', 'fr_id'])

ds_test_ids.set_transform(lambda _batch: {
  k: torch.tensor(v, dtype=torch.long, device=TORCH_DEV) for k, v in _batch.items()
}, columns=['eng_id', 'fr_id'])

ENG_VOCAB_SIZE = len(eng_tokenizer)
FR_VOCAB_SIZE = len(fr_tokenizer)
EMBEDDING_DIM = 128
MAX_EPOCHS = 100

train_dl = DataLoader(ds_train_ids, batch_size=32, shuffle=True)
test_dl = DataLoader(ds_test_ids, batch_size=32, shuffle=True)

encoder = Encoder(ENG_VOCAB_SIZE, EMBEDDING_DIM, 256, 2, 0.5, device=TORCH_DEV).to(TORCH_DEV)
decoder = Decoder(FR_VOCAB_SIZE, EMBEDDING_DIM, 256, 2, 0.5, token_start_of_sentence=eng_tokenizer.get_start_token_id(), max_sentence_len=SENTENCE_MAX_LEN, device=TORCH_DEV).to(TORCH_DEV)

model = SeqToSeq(encoder, decoder).to(TORCH_DEV)

loss_fn = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# summary(model)
# summary(encoder)
# summary(decoder)

def train_loop(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  model.train()
  for batch, e in enumerate(dataloader):
      X = e['eng_id']
      y = e['fr_id']

      optimizer.zero_grad()

      # Compute prediction and loss
      # pred = model(X, y)
      pred = model(X, y)
      loss = loss_fn(
        pred.view(-1, pred.size(-1)),
        y.view(-1)
      )

      # Backpropagation
      loss.backward()
      optimizer.step()

      if batch % 100 == 0:
          loss, current = loss.item(), (batch + 1) * len(X)
          print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
  model.eval()
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, correct = 0, 0

  with torch.no_grad():
      for e in dataloader:
          X = e['eng_id']
          y = e['fr_id']
          pred = model(X, None)
          test_loss += loss_fn(
            pred.view(-1, pred.size(-1)),
            y.view(-1)
          )
          correct += ( pred.view(-1, pred.size(-1)).argmax(1) == y.view(-1)).type(torch.float).sum().item()

  test_loss /= num_batches
  correct /= size * SENTENCE_MAX_LEN
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for t in range(MAX_EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dl, model, loss_fn, optimizer)
    test_loop(test_dl, model, loss_fn)
    torch.save(model.state_dict(), f'seq_to_seq_epoch_{t}.pt')
print("Done!")

