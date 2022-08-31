import os

from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


class InputFeature(object):
    def __init__(self, input_ids, input_mask, segment_ids, label, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label


class MyDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, item):
        return self.features[item]

    def __len__(self):
        return len(self.features)


def read_data(path):
    if not os.path.exists(path):
        print("Path {0} does not exist..\n", path)
    sentences = []
    labels = []
    with open(path, mode='r') as f:
        for line in f:
            items = line.split(',')
            sentences.append(items[0].strip())
            labels.append(items[1].strip())
    print("## Read finished, sentences length: {0}, tag length: {}\n", len(sentences), len(labels))
    return sentences, labels


def convert_example_to_feature(examples, tokenizer: BertTokenizer, batch_size, max_seq_lenth):
    features = []
    for sentence, label in examples:
        tokens = tokenizer.tokenize(sentence)
        if len(tokens) > max_seq_lenth - 2:
            tokens = tokens[:max_seq_lenth - 2]
        tokens.insert(0, "[CLS]")
        tokens.append("[SEP]")
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        if len(input_ids) < max_seq_lenth:
            sub_len = max_seq_lenth - len(input_ids)
            input_ids += [0] * sub_len
            token_type_ids += [0] * sub_len
            attention_mask += [0] * sub_len
        assert len(input_ids) == max_seq_lenth
        assert len(token_type_ids) == max_seq_lenth
        assert len(attention_mask) == max_seq_lenth
        features.append(InputFeature(input_ids, attention_mask, token_type_ids, label=label))
    return features
