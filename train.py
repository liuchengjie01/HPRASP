import argparse
import os

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from utils import MyDataset, convert_example_to_feature, read_data


def train(model, batch_size, max_seq_length, seed, data_path, epochs):
    print("## Train Process..")
    tokenizer = BertTokenizer.from_pretrained("models/bert-base-uncased")

    features = convert_example_to_feature(read_data(data_path), tokenizer, batch_size, max_seq_length)
    dataset = MyDataset(features)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        print("## In epoch {0}", epoch+1)
        for i, data in enumerate(tqdm(dataloader)):
            print("Length of batch: {0}\n", len(data))
            input_ids = []
            attention_mask = []
            token_type_ids = []
            labels = []
            for feature in data:
                input_ids.append(feature.input_ids)
                attention_mask.append(feature.input_mask)
                token_type_ids.append(feature.segment_ids)
                labels.append(feature.label)



if __name__ == '__main__':
    print("Hello!")
    parser = argparse.ArgumentParser();
    parser.add_argument("--model", type=str, required=False, default="bert")
    parser.add_argument("--path", type=str, required=False, default="data/SQLiV3.csv")
    parser.add_argument("--batch_size", type=int, required=False, default=64)
    parser.add_argument("--max_seq_len", type=int, required=False, default=32)
    parser.add_argument("--seed", type=int, required=False, default=1)
    parser.add_argument("--save_path", type=str, required=False, default="model/res")
    parser.add_argument("--epochs", type=int, required=False, default=5)
    args = parser.parse_args()
