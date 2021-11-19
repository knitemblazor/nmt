from dala_load import Multi30k
import torch
from model import Seq2SeqTransformer
import torch.nn as nn
import spacy
from torchtext.legacy.data import Field, BucketIterator, ReversibleField
from utilities import map_language, remap, save_src_vocab, save_tar_vocab, read_src_vocab, read_tar_vocab

UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok for tok in text.split() if tok.strip()]


SRC = Field(tokenize=tokenize_en,
            init_token='<sos>',
            pad_token='<pad>',
            eos_token='<eos>',
            lower=True)

TRG = Field(tokenize=tokenize_en,
            init_token='<sos>',
            pad_token='<pad>',
            eos_token='<eos>',
            lower=True)

train_data, valid_data, test_data = Multi30k.splits(exts=('.src', '.tar'),
                                                    fields=(SRC, TRG))

NUM_EPOCHS = 1000
batch_size= 5
SRC.build_vocab(train_data, min_freq=0)
TRG.build_vocab(train_data, min_freq=0)
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    device=DEVICE,shuffle=True)

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


torch.manual_seed(0)
src_vocab, src_len = read_src_vocab("SRC_vocab.txt")
trg_vocab, trg_len = read_tar_vocab("TRG_vocab.txt")
#
# SRC_VOCAB_SIZE = src_len +1
# TGT_VOCAB_SIZE = trg_len +1
SRC_VOCAB_SIZE = len(SRC.vocab)
TGT_VOCAB_SIZE = len(TRG.vocab)
EMB_SIZE = 512

NHEAD = 8
FFN_HID_DIM = 512
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


def train_epoch(model, optimizer, iterator):
    model.train()
    losses = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        tgt = batch.targ

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(iterator)


def evaluate(model, iterator):
    model.eval()
    losses = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        tgt = batch.targ
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[1:, :]
        # print("---:",tgt_out.reshape(-1))
        # print(logits)
        # if i == 100:
        #     exit()

        # print(logits.reshape(-1, logits.shape[-1]))
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(iterator)

from timeit import default_timer as timer


save_src_vocab(SRC.vocab, "SRC_vocab.txt")
save_tar_vocab(TRG.vocab, "TRG_vocab.txt")
import numpy as np
train_loss_list = []
# transformer.eval()
# transformer.load_state_dict(torch.load("model_19_nov.pth"))
for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer, train_iterator)
    train_loss_list.append(round(train_loss,3))
    patience = train_loss_list.count(min(train_loss_list))
    print(min(train_loss_list), patience)
    if patience == 400//batch_size:
    # if True:
        torch.save(transformer.state_dict(), 'model_19_nov.pth')
        break
    end_time = timer()
    val_loss = evaluate(transformer, valid_iterator)
    # print((f"Epoch: {epoch}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


def translate(inp_tex, mode):
    tex = "{} {}".format(mode,inp_tex).lower()
    # tex, code_list, lang_codes = map_language(tex)
    inp = [[src_vocab.get(i,0)] for i in tokenize_en(tex)]
    if len(inp)<25:
        for i in range(len(inp),25):
            inp.append([1])
    inp = torch.tensor(inp).view(-1, 1)
    src_mask = (torch.zeros(inp.shape[0], inp.shape[0])).type(torch.bool)
    tgt_tokens = greedy_decode(
                transformer,  inp, src_mask, max_len=inp.shape[0] + 4, start_symbol=SOS_IDX).flatten()
    return " ".join(trg_vocab.get(i.item()).strip() for i in tgt_tokens).replace("<sos>","").replace("<eos>","")


transformer.eval()
transformer.load_state_dict(torch.load("model_19_nov.pth"))
tar = open("machine_trans_data/train.tar", "r").read().splitlines()
src = open("machine_trans_data/train.src", "r").read().splitlines()
count = 0
for ind, line in enumerate(src):
    # input(("hi"))
    if tar[ind].strip() == translate(line, "").strip():
        count += 1
    # elif tar[ind].strip()!="tags are insufficient for text generation":
    else:
        print("{} {}".format(ind,line))
        print("  {}".format(tar[ind]))
        print("translated:", translate(line, "").strip())
        print("-----------------")
        # a=input("________________")
print(ind, count)
