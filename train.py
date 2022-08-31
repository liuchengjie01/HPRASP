import argparse

import numpy as np
import torch.nn.functional
import torch.nn.functional as F
from sklearn import metrics
from torch.optim import Adam
from tqdm import tqdm
from transformers import BertTokenizer

import utils
from bert import BertFilter
from utils import load_data


def train(model, config, tokenizer, dataloader, eval_dataloader):
    print("## Train Process..")

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = Adam(optimizer_grouped_parameters,
                     lr=config.learning_rate,
                     warmup=0.05,
                     t_total=len(dataloader) * config.epochs)
    dev_best_loss = float('inf')
    model.train()
    for epoch in range(config.epochs):
        print("## In epoch {0}", epoch + 1)
        cur_batch = 0
        for i, data in enumerate(tqdm(dataloader)):
            print("Length of batch: {0}\n", len(data))
            inputs = {}
            input_ids = []
            attention_mask = []
            token_type_ids = []
            labels = []
            for feature in data:
                input_ids.append(feature.input_ids)
                attention_mask.append(feature.input_mask)
                token_type_ids.append(feature.segment_ids)
                labels.append(feature.label)
            inputs["input_ids"] = input_ids
            inputs["attention_mask"] = attention_mask
            inputs["token_type_ids"] = token_type_ids
            # batch_size * num_labels
            outputs = model(inputs)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if cur_batch % 100 == 0:
                golds = labels.cpu()
                preds = torch.argmax(F.softmax(outputs))
                train_acc = metrics.accuracy_score(golds, preds)
                f1_score = metrics.f1_score(golds, preds)
                train_pre = metrics.precision_score(golds, preds)
                dev_acc, dev_loss = evaluate(model, config, tokenizer, eval_dataloader)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    model.save(model.state_dict(), config.save_path)
                    print("## save model state...\n")
                    print("Train Acc: {0}, Precision: {1}, F1_score: {2}..\n", train_acc, train_pre, f1_score)
            cur_batch += 1
            model.train()


def evaluate(model, eval_dataloader):
    print("## Eval...")
    model.eval()
    total_loss = 0
    total_labels = np.array([], dtype=int)
    total_preds = np.array([], dtype=int)
    with torch.no_grad():
        for eval_data in eval_dataloader:
            inputs = {}
            input_ids = []
            attention_mask = []
            token_type_ids = []
            labels = []
            for feature in eval_data:
                input_ids.append(feature.input_ids)
                attention_mask.append(feature.input_mask)
                token_type_ids.append(feature.segment_ids)
                labels.append(feature.label)
            inputs["input_ids"] = input_ids
            inputs["attention_mask"] = attention_mask
            inputs["token_type_ids"] = token_type_ids
            # batch_size * num_labels
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss
            total_labels = np.append(total_labels, labels.cpu().numpy())
            total_preds = np.append(total_preds, torch.argmax(F.softmax(outputs)))
    acc = metrics.accuracy_score(total_labels, total_preds)
    return acc, total_loss / len(eval_dataloader)


if __name__ == '__main__':
    print("Hello!")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False, default="bert")
    parser.add_argument("--path", type=str, required=False, default="data/SQLiV3.csv")
    parser.add_argument("--batch_size", type=int, required=False, default=64)
    parser.add_argument("--max_seq_len", type=int, required=False, default=32)
    parser.add_argument("--seed", type=int, required=False, default=1)
    parser.add_argument("--save_path", type=str, required=False, default="model/res")
    parser.add_argument("--epochs", type=int, required=False, default=5)
    args = parser.parse_args()

    config = utils.Config()

    model = BertFilter(config)

    tokenizer = BertTokenizer.from_pretrained(config.model_path)

    train_data, eval_data = load_data(config, tokenizer)

    train(model, config, tokenizer, train_data, eval_data)
