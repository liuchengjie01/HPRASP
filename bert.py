from torch import nn
from transformers import BertModel


class BertFilter(BertModel):
    def __init__(self, config):
        super(BertFilter, self).__init__()
        self.bert = BertModel.from_pretrained(config.model_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        pooler_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[1]
        out = self.fc(pooler_output)
        return out
