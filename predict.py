import torch
import re
from utilities import map_language, remap

UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tokenize_en(text):
    text = re.sub('\d', '', text)
    return [tok for tok in text.split() if tok.strip()]


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


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


def translate(transformer, src_vocab, trg_vocab,inp_tex, mode):
    tex = "{} {}".format(mode, inp_tex).lower()
    tex, code_list, lang_codes = map_language(tex)
    inp = [[src_vocab.get(i, 0)] for i in tokenize_en(tex)]
    if len(inp) < 25:
        for i in range(len(inp), 25):
            inp.append([1])
    inp = torch.tensor(inp).view(-1, 1)
    src_mask = (torch.zeros(inp.shape[0], inp.shape[0])).type(torch.bool)
    tgt_tokens = greedy_decode(
                transformer,  inp, src_mask, max_len=inp.shape[0] + 4, start_symbol=SOS_IDX).flatten()
    processed = " ".join(trg_vocab.get(i.item()).strip() for i in tgt_tokens).replace("<sos>","").replace("<eos>","")
    return remap(processed, code_list,lang_codes)


# a = translate("all documents documents equal to must be in language english","passed")
# print(a)