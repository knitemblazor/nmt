from lang_iso_map import lang_map


def map_language(text):
    lang_codes, lang_list = [], []
    for lang in lang_map:
        if lang in text.split():
            lang_list.append(lang)
            lang_codes.append(lang_map.get(lang))
            text = text.replace(lang, "langa")
    return text, lang_list, lang_codes


def remap(text, lang_list, lang_codes):
    count = 0
    processed_text = []
    for word in text.split():
        if "langc" == word.strip():
            if count < len(lang_list):
                word = lang_codes[count]
                count += 1
        if "langa" == word.strip():
            if count < len(lang_list):
                word = lang_list[count]

        processed_text.append(word)
    return " ".join(i for i in processed_text)


def save_src_vocab(vocab, path):
    with open(path, 'w+') as f:
        for token, index in vocab.stoi.items():
            f.write(f'{token}\t{index}\n')


def save_tar_vocab(vocab, path):
    with open(path, 'w+') as f:
        for token, index in vocab.stoi.items():
            f.write(f'{index}\t{token}\n')


def read_src_vocab(path):
    vocab = dict()
    with open(path, 'r') as f:
        for line in f:
            token, index = line.split('\t')
            vocab[token] = int(index)
    return vocab, int(index)


def read_tar_vocab(path):
    vocab = dict()
    with open(path, 'r') as f:
        for line in f:
            index,token = line.split('\t')
            vocab[int(index)] = token
    return vocab, int(index)