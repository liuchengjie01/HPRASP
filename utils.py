import os

import torch.cuda
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


class Config(object):
    def __init__(self):
        self.epochs = 5
        self.hidden_size = 768
        self.batch_size = 64
        self.max_seq_length = 32
        self.data_path = "data/train_SQLiV3.txt"
        self.eval_data = "data/dev_SQLiV3.txt"
        self.save_path = "models/save_state/bert_1.pth"
        self.model_path = "models/bert-base-uncased"
        self.seed = 1
        self.lr_rate = 1e-5
        self.device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
        self.num_labels = 2


def read_data(path):
    print("File path: {0}...", path)
    if not os.path.exists(path):
        print("Path {0} does not exist..\n", path)
        raise FileNotFoundError("Please check path: {0}...", path)
    sentences = []
    labels = []
    with open(path, mode='r', encoding='utf8') as f:
        for line in f:
            # print(line)
            items = line.split('\t')
            sentences.append(items[0].strip())
            try:
                labels.append(int(items[1].strip(), 10))
            except (ValueError, OSError):
                print("## Invalid Value: {0}...", items[1].strip())
                exit(-1)
    print("## Read finished, sentences length: {0}, tag length: {1}\n", len(sentences), len(labels))
    return sentences, labels


def convert_example_to_feature(examples, tokenizer: BertTokenizer, max_seq_length):
    features = []
    labels = examples[1]
    for i, sentence in enumerate(examples[0]):
        tokens = tokenizer.tokenize(sentence)
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:max_seq_length - 2]
        tokens.insert(0, "[CLS]")
        tokens.append("[SEP]")
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        if len(input_ids) < max_seq_length:
            sub_len = max_seq_length - len(input_ids)
            input_ids += [0] * sub_len
            token_type_ids += [0] * sub_len
            attention_mask += [0] * sub_len
        assert len(input_ids) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        features.append(InputFeature(input_ids, attention_mask, token_type_ids, label=labels[i]))
    return features


def collate_func(batch):
    out = dict()
    out["input_ids"] = torch.tensor([item.input_ids for item in batch], dtype=torch.long)
    out["attention_mask"] = torch.tensor([item.input_mask for item in batch], dtype=torch.long)
    out["token_type_ids"] = torch.tensor([item.segment_ids for item in batch],  dtype=torch.long)
    out["labels"] = torch.tensor([item.label for item in batch], dtype=torch.long)
    return out


def load_data(config, tokenizer):
    features = convert_example_to_feature(read_data(config.data_path), tokenizer, config.max_seq_length)
    dataset = MyDataset(features)
    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_func)

    eval_features = convert_example_to_feature(read_data(config.eval_data), tokenizer, config.max_seq_length)
    eval_dataset = MyDataset(eval_features)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=config.batch_size, collate_fn=collate_func)
    return dataloader, eval_dataloader
