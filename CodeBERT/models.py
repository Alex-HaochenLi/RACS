import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
# from transformers.modeling_bert import BertLayerNorm
# from transformers. transformers.models.bert.modeling_bert import BertLayerNorm
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
# from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
#                           BertConfig, BertForMaskedLM, BertTokenizer,
#                           GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
#                           OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
#                           RobertaConfig, RobertaModel, RobertaTokenizer,
#                           DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
from transformers.modeling_utils import PreTrainedModel


class ModelContra(PreTrainedModel):
    def __init__(self, encoder, config, tokenizer, args):
        super(ModelContra, self).__init__(config)
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer

        # self.loss_func = nn.BCELoss()
        self.loss_func = CrossEntropyLoss()
        self.args = args

        self.uniform = torch.distributions.uniform.Uniform(0.90, 1.10)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, code_inputs, nl_inputs, labels=None, code_lens=None, return_vec=False):
        bs = code_inputs.shape[0]
        inputs = torch.cat((code_inputs, nl_inputs), 0)
        outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[1]
        code_vec_ori = outputs[:bs]
        nl_vec_ori = outputs[bs:]
        if return_vec:
            return code_vec_ori, nl_vec_ori

        if self.args.ra:
            aug_index = torch.randint(0, 4, (1,))
            if aug_index == 0:
                nl_vec, code_vec, labels = self.generate_linear_interpolate_data(code_vec_ori, nl_vec_ori, labels)
            elif aug_index == 1:
                nl_vec, code_vec, labels = self.generate_binary_interpolate_data(code_vec_ori, nl_vec_ori, labels)
            elif aug_index == 2:
                nl_vec, code_vec, labels = self.generate_dropout_data(code_vec_ori, nl_vec_ori, labels)
            elif aug_index == 3:
                nl_vec, code_vec, labels = self.generate_scaling_data(code_vec_ori, nl_vec_ori, labels)
        else:
            nl_vec = nl_vec_ori
            code_vec = code_vec_ori

        logits = torch.matmul(nl_vec, code_vec.T)

        if self.args.ra:
            matrix_labels = torch.diag(labels).float()  # (Batch, Batch)

            matrix_labels = self.generate_soft_labels(matrix_labels, batch_size=code_vec_ori.size(0), mix_time=self.args.mix_time)

            neg_index = (matrix_labels == 0).nonzero().view(-1, (bs - 1) * (self.args.mix_time + 1), 2).repeat(1, self.args.mix_time + 1, 1).view(-1, (bs - 1) * (self.args.mix_time + 1), 2)
            neg_index = neg_index.view(-1, 2)
            neg_logits = logits[neg_index[:, 0], neg_index[:, 1]].view(-1, (bs - 1) * (self.args.mix_time + 1))
            pos_index = (matrix_labels == 1).nonzero()
            pos_logits = logits[pos_index[:, 0], pos_index[:, 1]].unsqueeze(1)
            logits = torch.cat([pos_logits, neg_logits], dim=1)

        if self.args.ra:
            loss = self.loss_func(logits, torch.zeros(logits.size(0), device=logits.device).long())
        else:
            loss = self.loss_func(logits, torch.arange(logits.size(0), device=logits.device).long())

        predictions = None
        return loss, predictions

    def generate_soft_labels(self, matrix_labels, batch_size, mix_time):

        for i in range(mix_time):
            matrix_labels = matrix_labels + torch.diag(torch.ones((mix_time - i) * batch_size, device=matrix_labels.device), batch_size * (i + 1))
        matrix_labels = matrix_labels + torch.triu(matrix_labels, diagonal=1).T

        return matrix_labels

    def generate_indices(self, batchsize, mix_time):
        indices = torch.zeros(mix_time * batchsize).to(self.args.device)
        for i in range(mix_time):
            while True:
                index = torch.randperm(batchsize).to(self.args.device)
                if not (index == torch.arange(index.size(0)).to(self.args.device)).any():
                    indices[i * batchsize: (i + 1) * batchsize] = index
                    break

        return indices.long()

    def generate_linear_interpolate_data(self, code_vec_ori, nl_vec_ori, labels):
        l = self.uniform.sample((self.args.mix_time * code_vec_ori.size(0), 1)).to(code_vec_ori.device)

        index = self.generate_indices(code_vec_ori.size(0), self.args.mix_time)
        rand_code, rand_nl = code_vec_ori[index], nl_vec_ori[index]

        code_vec = torch.cat([code_vec_ori] * self.args.mix_time, dim=0)
        nl_vec = torch.cat([nl_vec_ori] * self.args.mix_time, dim=0)
        combined_code = l * code_vec + (1 - l) * rand_code
        combined_nl = l * nl_vec + (1 - l) * rand_nl

        nl_vec = torch.cat([nl_vec_ori, combined_nl], dim=0)
        code_vec = torch.cat([code_vec_ori, combined_code], dim=0)
        labels = torch.cat([labels, torch.ones(combined_code.size(0)).to(self.args.device)], dim=0)

        return nl_vec, code_vec, labels

    def generate_binary_interpolate_data(self, code_vec_ori, nl_vec_ori, labels):
        l = 0.25
        tmp = self.uniform.sample((1, code_vec_ori.size(1))).to(self.args.device)
        pos_mask = (tmp > l).int()
        neg_mask = (~ (tmp > l)).int()

        nl_index = self.generate_indices(code_vec_ori.size(0), self.args.mix_time)
        code_index = self.generate_indices(code_vec_ori.size(0), self.args.mix_time)
        rand_code, rand_nl = code_vec_ori[code_index], nl_vec_ori[nl_index]

        code_vec = torch.cat([code_vec_ori] * self.args.mix_time, dim=0)
        nl_vec = torch.cat([nl_vec_ori] * self.args.mix_time, dim=0)
        combined_code = pos_mask * code_vec + neg_mask * rand_code
        combined_nl = pos_mask * nl_vec + neg_mask * rand_nl

        nl_vec = torch.cat([nl_vec_ori, combined_nl], dim=0)
        code_vec = torch.cat([code_vec_ori, combined_code], dim=0)
        labels = torch.cat([labels, torch.ones(combined_code.size(0)).to(self.args.device)], dim=0)

        return nl_vec, code_vec, labels

    def generate_dropout_data(self, code_vec_ori, nl_vec_ori, labels):
        code_vec = torch.cat([code_vec_ori] * self.args.mix_time, dim=0)
        nl_vec = torch.cat([nl_vec_ori] * self.args.mix_time, dim=0)

        combined_code = self.dropout(code_vec)
        combined_nl = self.dropout(nl_vec)

        nl_vec = torch.cat([nl_vec_ori, combined_nl], dim=0)
        code_vec = torch.cat([code_vec_ori, combined_code], dim=0)
        labels = torch.cat([labels, torch.ones(combined_code.size(0)).to(self.args.device)], dim=0)

        return nl_vec, code_vec, labels

    def generate_scaling_data(self, code_vec_ori, nl_vec_ori, labels):
        code_vec = torch.cat([code_vec_ori] * self.args.mix_time, dim=0)
        nl_vec = torch.cat([nl_vec_ori] * self.args.mix_time, dim=0)

        combined_code = torch.normal(1, 0.1, (code_vec.size(0), code_vec.size(1)), device=code_vec.device) * code_vec
        combined_nl = torch.normal(1, 0.1, (nl_vec.size(0), nl_vec.size(1)), device=nl_vec.device) * nl_vec

        nl_vec = torch.cat([nl_vec_ori, combined_nl], dim=0)
        code_vec = torch.cat([code_vec_ori, combined_code], dim=0)
        labels = torch.cat([labels, torch.ones(combined_code.size(0)).to(self.args.device)], dim=0)

        return nl_vec, code_vec, labels
