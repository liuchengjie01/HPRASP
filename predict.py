import argparse

import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from bert import BertFilter
from utils import Config


def pre_load(config):
    model = BertFilter(config)
    model.load_state_dict(torch.load(config.save_path))
    tokenizer = BertTokenizer.from_pretrained(config.model_path)
    return model, tokenizer


def test():
    config = Config()
    tokenizer = BertTokenizer.from_pretrained(config.model_path)
    tokens = tokenizer.tokenize("Hello world")
    print(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(input_ids)


def predict(model, tokenizer, sent):
    inputs = tokenizer.encode_plus(sent, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
    print(inputs)
    model.eval()
    output = model(**inputs)
    result = torch.argmax(F.softmax(output, dim=1), dim=1)
    result = result.cpu().numpy()
    print(inputs)
    print("## Classification Result: {0}".format(result))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--sent", type=str, required=True, default="SeLEct * FROM table Name")
    #
    # args = parser.parse_args()
    # config = Config()
    # model, tokenizer = pre_load(config=config)
    # predict(model, tokenizer, args.sent)
    test()
