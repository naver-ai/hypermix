# Code adapted from
# https://github.com/GT-SALT/MixText/blob/master/code/mixtext.py

__all__ = ["TMixBertModel", "BertEncoder4Mix",
           "TMixBertForSequenceClassification", "TMixClassifier"]

import random
from typing import Sequence

import torch
import torch.nn as nn
import transformers

from dataset import Example
from .classifier import TransformersClassifier


class BertEncoder4Mix(nn.Module):
    def __init__(self, config):
        super(BertEncoder4Mix, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([transformers.BertLayer(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, hidden_states2=None, l=None,
                mix_layer=1000, attention_mask=None, attention_mask2=None,
                head_mask=None):
        all_hidden_states = ()
        all_attentions = ()

        # Perform mix at till the mix_layer
        if mix_layer == -1:
            if hidden_states2 is not None:
                hidden_states = l * hidden_states + (1 - l) * hidden_states2

        for i, layer_module in enumerate(self.layer):
            if i <= mix_layer:

                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

                if hidden_states2 is not None:
                    layer_outputs2 = layer_module(
                        hidden_states2, attention_mask2, head_mask[i])
                    hidden_states2 = layer_outputs2[0]

            if i == mix_layer:
                if hidden_states2 is not None:
                    l_ = l.unsqueeze(-1).unsqueeze(-1)
                    hidden_states = (l_ * hidden_states +
                                     (1 - l_) * hidden_states2)

            if i > mix_layer:
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # last-layer hidden state, (all hidden states), (all attentions)
        return outputs


class TMixBertModel(transformers.BertPreTrainedModel):
    def __init__(self, config):
        super(TMixBertModel, self).__init__(config)
        self.embeddings = transformers.models.bert.modeling_bert.BertEmbeddings(
            config)
        self.encoder = BertEncoder4Mix(config)
        self.pooler = transformers.models.bert.modeling_bert.BertPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(
            old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, mix_idx=None, l=None, mix_layer=1000,
                attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None):

        input_ids2 = None
        if mix_idx is not None:
            input_ids2 = input_ids[mix_idx]

        attention_mask2 = None
        if attention_mask is None:
            if input_ids2 is not None:
                attention_mask2 = torch.ones_like(input_ids2)
            attention_mask = torch.ones_like(input_ids)
        elif mix_idx is not None:
            attention_mask2 = attention_mask[mix_idx]

        token_type_ids2 = None
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            if input_ids2 is not None:
                token_type_ids2 = torch.zeros_like(input_ids2)
        elif mix_idx is not None:
            token_type_ids2 = token_type_ids[mix_idx]

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        extended_attention_mask2 = None
        if input_ids2 is not None:
            extended_attention_mask2 = attention_mask2.unsqueeze(
                1).unsqueeze(2)

            extended_attention_mask2 = extended_attention_mask2.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask2 = ((1.0 - extended_attention_mask2)
                                        * -10000.0)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(
                    0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                # We can specify head_mask for each layer
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            # switch to fload if need + fp16 compatibility
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        embedding_output2 = None
        if input_ids2 is not None:
            embedding_output2 = self.embeddings(
                input_ids2, position_ids=position_ids,
                token_type_ids=token_type_ids2)

        if input_ids2 is not None:
            encoder_outputs = self.encoder(embedding_output, embedding_output2,
                                           l, mix_layer,
                                           extended_attention_mask,
                                           extended_attention_mask2,
                                           head_mask=head_mask)
        else:
            encoder_outputs = self.encoder(
                embedding_output, attention_mask=extended_attention_mask,
                head_mask=head_mask)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs


class TMixBertForSequenceClassification(transformers.BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = TMixBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            mix_idx=None,
            l=None,
            mix_layer=1000,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs = self.bert(
            input_ids,
            mix_idx=mix_idx,
            l=l,
            mix_layer=mix_layer,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        output = (logits,) + outputs[2:]
        return output


class TMixClassifier(TransformersClassifier):

    def __init__(self, *args, alpha=0.75,
                 mix_layer_set=frozenset((7, 9, 12)), **kwargs):
        super().__init__(
            *args,
            model_cls=TMixBertForSequenceClassification,
            **kwargs
        )
        self.alpha = alpha
        self.beta_dist = torch.distributions.Beta(self.alpha, self.alpha)
        self.mix_layer_set = mix_layer_set

    def compute_forward(self, texts: Sequence[str]):
        x = self.tokenizer.batch_encode_plus(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return self.model(**{k: v.to(self.device) for k, v in x.items()})[0]

    def compute_loss(self, batch: Sequence[Example]):
        batch_size = len(batch)
        mix_factor = self.beta_dist.sample([batch_size]).to(self.device)
        rand_idx = torch.randperm(batch_size).to(self.device)
        batch_dict = self.tokenizer.batch_encode_plus(
            [ex.text for ex in batch],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        target = self.create_label_tensor(batch)
        target_other = target[rand_idx]
        target_mix = (target * mix_factor.unsqueeze(-1) +
                      target_other * mix_factor.unsqueeze(-1))
        logit = self.model(
            mix_idx=rand_idx,
            l=mix_factor,
            mix_layer=random.choice(list(self.mix_layer_set)),
            **{k: v.to(self.device) for k, v in batch_dict.items()}
        )[0]
        return self.loss_fn(logit, target_mix)
