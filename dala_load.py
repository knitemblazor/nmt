import os
import io

import torchtext.legacy.data as data


class TranslationDataset(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.targ))

    def __init__(self, path, exts, fields, **kwargs):
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('targ', fields[1])]

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        print(src_path,trg_path)
        with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
                io.open(trg_path, mode='r', encoding='utf-8') as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line], fields))

        super(TranslationDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, exts, fields, path="", root='machine_trans_data',
               train='train', validation='val', test='test', **kwargs):

        train_data = cls(
            os.path.join(path, train), exts, fields, **kwargs)
        val_data = cls(
            os.path.join(path, validation), exts, fields, **kwargs)
        test_data = cls(
            os.path.join(path, test), exts, fields, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


class Multi30k(TranslationDataset):

    # name = 'multi30k'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='machine_trans_data',
               train='train', validation='val', test='test', **kwargs):

        if 'path' not in kwargs:
            expected_folder = root
            path = expected_folder if os.path.exists(expected_folder) else None
        else:
            path = kwargs['path']
            del kwargs['path']

        return super(Multi30k, cls).splits(
            exts, fields, path, root, train, validation, test, **kwargs)
