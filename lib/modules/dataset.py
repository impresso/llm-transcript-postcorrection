import torch
import pandas as pd
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset

"""
Dataset classes for all text correction datasets
"""


class OCRGTDataset(Dataset):

    def __init__(self, filename):
        pass


class OverproofDataset(Dataset):

    def __init__(self, filename):
        pass


class ImpressoFrakturDataset(Dataset):

    def __init__(self, filename):
        pass


class ImpressoHIPEDataset(Dataset):

    def __init__(self, filename):
        pass


class RETASDataset(Dataset):

    def __init__(self, filename):
        pass


class OCR19thSACDataset(Dataset):

    def __init__(self, filename):
        pass


class DBNLDataset(Dataset):

    def __init__(self, filename):
        pass


class ICDARDataset(Dataset):

    def __init__(self, filename):
        pass


class AJMCDataset(Dataset):

    def __init__(self, filename):
        pass


def _read_conll(path, encoding='utf-8', sep=None, indexes=None, dropna=True):
    r"""
    Construct a generator to read conll items.
    :param path: file path
    :param encoding: file's encoding, default: utf-8
    :param sep: seperator
    :param indexes: conll object's column indexes that needed, if None, all columns are needed. default: None
    :param dropna: weather to ignore and drop invalid data,
            :if False, raise ValueError when reading invalid data. default: True
    :return: generator, every time yield (line number, conll item)
    """

    def parse_conll(sample):

        sample = list(map(list, zip(*sample)))
        sample = [sample[i] for i in indexes]

        for f in sample:
            if len(f) <= 0:
                raise ValueError('empty field')
        return sample

    with open(path, 'r', encoding=encoding) as f:

        sample = []
        start = next(f).strip()  # Skip columns
        start = next(f).strip()

        data = []
        for line_idx, line in enumerate(f, 0):
            line = line.strip()

            if any(
                substring in line for substring in [
                    'DOCSTART',
                    '###',
                    "# id",
                    "# ",
                    '###']):
                continue

            if line == '':
                if len(sample):
                    try:
                        res = parse_conll(sample)
                        sample = []
                        if ['TOKEN'] not in res:
                            if ['Token'] not in res:
                                data.append([line_idx, res])
                    except Exception as e:
                        if dropna:
                            print(
                                'Invalid instance which ends at line: {} has been dropped.'.format(line_idx))
                            sample = []
                            raise e
                        raise ValueError(
                            'Invalid instance which ends at line: {}'.format(line_idx))
            elif 'EndOfSentence' in line:
                sample.append(
                    line.split(sep)) if sep else sample.append(
                    line.split())

                if len(sample):
                    try:
                        res = parse_conll(sample)
                        sample = []
                        if ['TOKEN'] not in res:
                            if ['Token'] not in res:
                                data.append([line_idx, res])
                    except Exception as e:
                        if dropna:
                            print(
                                'Invalid instance which ends at line: {} has been dropped.'.format(line_idx))
                            sample = []
                            raise e
                        raise ValueError(
                            'Invalid instance which ends at line: {}'.format(line_idx))
            else:
                sample.append(
                    line.split(sep)) if sep else sample.append(
                    line.split())

        if len(sample) > 0:
            try:
                res = parse_conll(sample)
                if ['TOKEN'] not in res:
                    if ['Token'] not in res:
                        data.append([line_idx, res])
            except Exception as e:
                if dropna:
                    return
                print('Invalid instance ends at line: {}'.format(line_idx))
                raise e

        return data


class OCRDataset(Dataset):

    def __init__(self, dataset, tokenizer, max_len, test_set=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.test_set = test_set
        if self.test_set:
            df = pd.read_csv(dataset, sep="\t", names=["id", "text"])
            self.classes = None
            self.encoded_classes = None
        else:
            df = pd.read_csv(dataset, sep="\t", names=["id", "text", "labels"])

            self.encoded_classes = pd.unique(df['labels'])
            self.targets = df['label'].to_numpy()

        self.sequences = df['text'].to_numpy()
        self.df = df

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = str(self.sequences[index])
        if not self.test_set:
            target = self.targets[index]

        encoding = self.tokenizer.encode_plus(
            sequence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        if self.test_set:
            return {
                'sequence': sequence,
                'input_ids': torch.tensor(
                    encoding['input_ids'],
                    dtype=torch.long),
                'attention_mask': torch.tensor(
                    encoding['attention_mask'],
                    dtype=torch.long)}
        else:
            return {
                'sequence': sequence, 'input_ids': torch.tensor(
                    encoding['input_ids'], dtype=torch.long), 'attention_mask': torch.tensor(
                    encoding['attention_mask'], dtype=torch.long), 'target': torch.tensor(
                    target, dtype=torch.long)}

    def get_info(self):
        return self.classes, self.encoded_classes, self.df.shape

    def get_dataframe(self):
        return self.df


class NERDataset(TorchDataset):

    def __init__(self, filename):

        indexes = list(range(10))  # -3 is for EL
        columns = ["TOKEN", "NE-COARSE-LIT", "NE-COARSE-METO", "NE-FINE-LIT",
                   "NE-FINE-METO", "NE-FINE-COMP", "NE-NESTED",
                   "NEL-LIT", "NEL-METO", "MISC"]
        self.phrases = _read_conll(
            filename,
            encoding='utf-8',
            sep='\t',
            indexes=indexes,
            dropna=True)

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, index):
        phrase = str(self.phrases[index])

        return phrase

    def get_info(self):
        return self.phrases

    def get_dataframe(self):
        return self.phrases
