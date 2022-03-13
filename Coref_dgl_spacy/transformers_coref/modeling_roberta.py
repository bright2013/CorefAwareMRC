# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RoBERTa model. """

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging

import dgl
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from .modeling_bert import BertEmbeddings, BertLayerNorm, BertModel, BertPreTrainedModel, gelu, BertConfig, BertLayer
from .configuration_roberta import RobertaConfig
from .file_utils import add_start_docstrings
import numpy as np

logger = logging.getLogger(__name__)

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    'distilroberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    'roberta-base-openai-detector': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    'roberta-large-openai-detector': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
    'coref-roberta-large': "https://cloud.tsinghua.edu.cn/d/a2eed603edd949b0a663/files/?p=%2FCorefRoBERTa-large%2Fpytorch_model.bin&dl=1",
    'coref-roberta-base': "https://cloud.tsinghua.edu.cn/d/a2eed603edd949b0a663/files/?p=%2FCorefRoBERTa-base%2Fpytorch_model.bin&dl=1",
}


class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super(RobertaEmbeddings, self).__init__(config)
        self.padding_idx = 1
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size,
                                                padding_idx=self.padding_idx)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is None:
            # Position numbers begin at padding_idx+1. Padding symbols are ignored.
            # cf. fairseq's `utils.make_positions`
            position_ids = torch.arange(self.padding_idx + 1, seq_length + self.padding_idx + 1, dtype=torch.long,
                                        device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        return super(RobertaEmbeddings, self).forward(input_ids,
                                                      token_type_ids=token_type_ids,
                                                      position_ids=position_ids,
                                                      inputs_embeds=inputs_embeds)


ROBERTA_START_DOCSTRING = r"""    The RoBERTa model was proposed in
    `RoBERTa: A Robustly Optimized BERT Pretraining Approach`_
    by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer,
    Veselin Stoyanov. It is based on Google's BERT model released in 2018.
    
    It builds on BERT and modifies key hyperparameters, removing the next-sentence pretraining
    objective and training with much larger mini-batches and learning rates.
    
    This implementation is the same as BertModel with a tiny embeddings tweak as well as a setup for Roberta pretrained 
    models.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`RoBERTa: A Robustly Optimized BERT Pretraining Approach`:
        https://arxiv.org/abs/1907.11692

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~transformers.RobertaConfig`): Model configuration class with all the parameters of the 
            model. Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

ROBERTA_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, RoBERTa input sequence should be formatted with <s> and </s> tokens as follows:

            (a) For sequence pairs:

                ``tokens:         <s> Is this Jacksonville ? </s> </s> No it is not . </s>``

            (b) For single sequences:

                ``tokens:         <s> the dog is hairy . </s>``

            Fully encoded sequences or sequence pairs can be obtained using the RobertaTokenizer.encode function with 
            the ``add_special_tokens`` parameter set to ``True``.

            RoBERTa is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.

            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional` need to be trained) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Optional segment token indices to indicate first and second portions of the inputs.
            This embedding matrice is not trained (not pretrained during RoBERTa pretraining), you will have to train it
            during finetuning.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            (see `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_ for more details).
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1[``.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **inputs_embeds**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, embedding_dim)``:
            Optionally, instead of passing ``input_ids`` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""


@add_start_docstrings(
    "The bare RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING)
class RobertaModel(BertModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaModel, self).__init__(config)

        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


@add_start_docstrings("""RoBERTa Model with a `language modeling` head on top. """,
                      ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING)
class RobertaForMaskedLM(BertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMaskedLM.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForMaskedLM, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None,
                masked_lm_labels=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super(RobertaLMHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x) + self.bias

        return x


@add_start_docstrings("""RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer 
    on top of the pooled output) e.g. for GLUE tasks. """,
                      ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING)
class RobertaForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None,
                labels=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


@add_start_docstrings("""Roberta Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
                      ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING)
class RobertaForMultipleChoice(BertPreTrainedModel):
    r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            To match pre-training, RoBerta input sequence should be formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] [SEP] no it is not . [SEP]``

                ``token_type_ids:   0   0  0    0    0     0       0   0   0     1  1  1  1   1   1``

            (b) For single sequences:

                ``tokens:         [CLS] the dog is hairy . [SEP]``

                ``token_type_ids:   0   0   0   0  0     0   0``

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **inputs_embeds**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, embedding_dim)``:
            Optionally, instead of passing ``input_ids`` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **classification_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above).
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMultipleChoice.from_pretrained('roberta-base')
        choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]
        input_ids = torch.tensor([tokenizer.encode(s, add_special_tokens=True) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForMultipleChoice, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, inputs_embeds=None):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.roberta(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids,
                               attention_mask=flat_attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


@add_start_docstrings("""Roberta Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
                      ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING)
class RobertaForTokenClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForTokenClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForQuestionAnswering(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            answer_masks=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    Examples::
        # The checkpoint roberta-large is not fine-tuned for question answering. Please see the
        # examples/run_squad.py example to see how to fine-tune a model to a question answering task.
        from transformers import RobertaTokenizer, RobertaForQuestionAnswering
        import torch
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForQuestionAnswering.from_pretrained('roberta-base')
        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_ids = tokenizer.encode(question, text)
        start_scores, end_scores = model(torch.tensor([input_ids]))
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
        """

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduce=False)

            start_losses = [(loss_fct(start_logits, _start_positions) * _span_mask) \
                            for (_start_positions, _span_mask) \
                            in zip(torch.unbind(start_positions, dim=1), torch.unbind(answer_masks, dim=1))]
            end_losses = [(loss_fct(end_logits, _end_positions) * _span_mask) \
                          for (_end_positions, _span_mask) \
                          in zip(torch.unbind(end_positions, dim=1), torch.unbind(answer_masks, dim=1))]

            total_loss = sum(start_losses + end_losses)
            total_loss = torch.mean(total_loss) / 2

            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)



def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        #mask = mask.half()

        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result

class SCAttention(nn.Module) :
    def __init__(self, input_size, hidden_size) :
        super(SCAttention, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size, hidden_size)
        self.map_linear = nn.Linear(hidden_size, hidden_size)
        #self.map_linear = nn.Linear(4 * hidden_size, hidden_size)
        #self.map_linear = nn.Linear(4 * hidden_size, hidden_size)
        self.init_weights()

    def init_weights(self) :
        nn.init.xavier_uniform_(self.W.weight.data)
        self.W.bias.data.fill_(0.1)

    #W_wq_relu_cox4_nolastRelu
    def forward(self, passage, p_mask, question, q_mask):
        Wp = F.relu(self.W(passage))
        Wq = F.relu(self.W(question))
        scores = torch.bmm(Wp, Wq.transpose(2, 1))
        # mask = q_mask.unsqueeze(1).repeat(1, passage.size(1), 1)
        q_mask = torch.transpose(q_mask, 1, 2)
        mask = q_mask.repeat(1, passage.size(1), 1)
        alpha = masked_softmax(scores, mask)
        output = torch.bmm(alpha, Wq)
        output = self.map_linear(output)
        return output

# Copied from transformers.models.bert.modeling_bert.BertPooler
class RobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class RobertaForQuestionAnsweringDGL(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_answers = config.num_answers
        self.hidden_size = config.hidden_size

        self.roberta = RobertaModel(config)
        # change first param to 2 * config.hidden_size
        self.qa_outputs = nn.Linear(2 * config.hidden_size, self.num_labels)
        self.qa_classifier = nn.Linear(config.hidden_size, self.num_answers)
        self.GCN = RGCNModel(config.hidden_size, 2, 1, True)
        # self.GCN.to(self.device)
        self.down_input = nn.Linear(config.hidden_size, 128)

        self.proj_attention = SCAttention(config.hidden_size, config.hidden_size)
        self.pooler = RobertaPooler(config)

        self.reduce_layer = nn.Linear(1024, 32)
        self.expand_layer = nn.Linear(32, 1024)

        self.point_outputs = nn.Linear(config.hidden_size, 1)
        # BERT large has 16 attention heads with 340 million parameters //BERT base has a total of 12 attention heads
        head_num = config.num_attention_heads // 4
        # print(head_num)
        # self.roberta = RobertaModel(config)
        # self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            answer_masks=None,
            answer_nums=None,
            doc_coref=None
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    Examples::
        # The checkpoint roberta-large is not fine-tuned for question answering. Please see the
        # examples/run_squad.py example to see how to fine-tune a model to a question answering task.
        from transformers import RobertaTokenizer, RobertaForQuestionAnswering
        import torch
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForQuestionAnswering.from_pretrained('roberta-base')
        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_ids = tokenizer.encode(question, text)
        start_scores, end_scores = model(torch.tensor([input_ids]))
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
        """





        device = input_ids.device if input_ids is not None else inputs_embeds.device


        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]


        #perform dgl, got coref matrix of same size and then form graph
        batch_id = 0
        batch_coref_ids = []
        batch_size = sequence_output.size(0)
        seq_len = sequence_output.size(1)
        hidden_size = sequence_output.size(2)
        batch_graph_results = torch.ones(batch_size, seq_len, hidden_size).to(device)

        coref_rep = []
        coref_mask = []
        for batchi in range(batch_size):
            arr = doc_coref[batchi]
            max_value = max(arr)
            g = dgl.DGLGraph().to(device)

            reduced_sequence_output = self.reduce_layer(sequence_output[batchi])
            g.add_nodes(len(doc_coref[batchi]))
            # for coref_value in range(1, max_value + 1):
            #     coref_array = torch.where(arr == coref_value)
            #     for i in range(len(coref_array[0])):
            #         src = torch.LongTensor([coref_array[0][i]] * len(coref_array[0])).to(device)
            #         dst = torch.LongTensor(list(coref_array[0])).to(device)
            #         g.add_edges(src, dst)
            #
            #
            # # add node features
            # # print(sequence_output[batchi].shape)
            # # print(sequence_output[batchi])
            # indexes = torch.LongTensor(range(len(doc_coref[batchi]))).to(device)
            # g.nodes[range(len(doc_coref[batchi]))].data['h'] = torch.index_select(sequence_output[batchi], 0, indexes)


            for coref_value in range(1, max_value + 1):
                coref_array = torch.where(arr == coref_value)
                for i in range(len(coref_array[0])):
                    src = torch.LongTensor([coref_array[0][i]] * len(coref_array[0])).to(device)
                    dst = torch.LongTensor(list(coref_array[0])).to(device)
                    g.add_edges(src, dst)
            nodes_to_remove = []
            nodes_selected = []
            # nodes_selected.append(0)
            # add global edges
            for x in range(1, len(arr)):
                if arr[x] >0:
                    g.add_edges(0, x)
                    nodes_selected.append(x)
                else:
                    nodes_to_remove.append(x)

            g.nodes[torch.LongTensor(range(len(arr))).to(device)].data['h'] = torch.index_select(reduced_sequence_output, 0, torch.LongTensor(range(len(arr))).to(device))
            # g.remove_nodes(nodes_to_remove)
            # print("node 0 features", len(g.ndata['h'][0]))
            # print("seq features", len(sequence_output[batchi][0]))
            edge_norm = []
            edge_types = []
            edges = []




            # for i in range(len(g.edges[0])):
            #     edges.append([g.edges[0][i], g.edges[1][i]])

            # add global edges
            # for x in range(1, len(doc_coref[batchi])):
            #     g.add_edges(0, x)
                # edge_types.append(2)
                # edges.append([0, x])
            # g = dgl.add_reverse_edges(g)
            # print(g.edges())
            for i in range(len(g.edges()[0])):
                edges.append([g.edges()[0][i], g.edges()[1][i]])

            for e1, e2 in edges:
                if e1 == e2:
                    edge_norm.append(1)
                    edge_types.append(1)
                elif g.in_degrees(e2) == 1:
                    edge_norm.append(1)
                    edge_types.append(2)
                else:
                    edge_norm.append(1 / (g.in_degrees(e2) - 1))
                    edge_types.append(1)
            #
            # for e1, e2 in g.edges:
            #     if e1 == e2:
            #         edge_norm.append(1)
            #         edge_types.append(1)
            #     else:
            #         edge_norm.append(1 / (g.in_degrees(e2) - 1))
            #         edge_types.append(1)

            edge_type = torch.LongTensor(edge_types).to(device)
            edge_norm = torch.Tensor(edge_norm).unsqueeze(1).float().to(device)
            g.edata.update({'rel_type': edge_type, })
            g.edata.update({'norm': edge_norm})
            # print(f"edges {len(g.edges()[0])}")
            # print(f"nodes {len(g.nodes())}")
            X = self.GCN(g)[0]
            expanded_x = self.expand_layer(X)
            # print(f"expanded_x shape: {expanded_x.shape}")
            # coref_rep.append(X)
            # coref_mask.append(torch.tensor([1] * X.size(0), dtype=torch.bool))


            batch_graph_results[batchi,:,:] = expanded_x[:,:]
        # print(f"coref_rep before: {len(coref_rep)}")
        # coref_rep = torch.nn.utils.rnn.pad_sequence(coref_rep).to(device)
        # coref_rep = torch.transpose(coref_rep, 0, 1).contiguous()
        # print(f"coref_rep after: {coref_rep.shape}")
        # print(f"coref_mask before: {len(coref_mask)}")
        #
        # coref_mask = torch.nn.utils.rnn.pad_sequence(coref_mask).to(device).unsqueeze(2)
        # coref_mask = torch.transpose(coref_mask, 0, 1).contiguous()
        # print(f"coref_mask after: {coref_mask.shape}")
        #
        # #     # final_rep = self.logic_op(svo_rep, svo_mask)
        # output = self.proj_attention(sequence_output, flat_attention_mask.unsqueeze(2), coref_rep, coref_mask)
        # # final_rep = self.pooler(output)
        # print(output.shape)
        # print(sequence_output.shape)

        # print(sequence_output.shape)
        # print(batch_graph_results.shape)
        # perform fuse
        with_gcn = torch.cat((sequence_output, batch_graph_results), 2)
        # print("fused gcn shape:", with_gcn.shape)
        # print("hidden size:",  str(self.config.hidden_size))



        logits = self.qa_outputs(with_gcn)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        switch_logits = self.qa_classifier(sequence_output[:, 0, :])

        outputs = (start_logits, end_logits, switch_logits) + outputs[2:]

        if start_positions is not None and end_positions is not None:

            if len(answer_nums.size()) > 1:
                answer_nums = answer_nums.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduce=False)

            start_losses = [(loss_fct(start_logits, _start_positions) * _span_mask) \
                            for (_start_positions, _span_mask) \
                            in zip(torch.unbind(start_positions, dim=1), torch.unbind(answer_masks, dim=1))]
            end_losses = [(loss_fct(end_logits, _end_positions) * _span_mask) \
                          for (_end_positions, _span_mask) \
                          in zip(torch.unbind(end_positions, dim=1), torch.unbind(answer_masks, dim=1))]

            s_e_loss = sum(start_losses + end_losses)  # bsz
            switch_loss = loss_fct(switch_logits, answer_nums)

            total_loss = torch.mean(s_e_loss + switch_loss)  # + 0.1 * mention_loss

            outputs = (total_loss,) + outputs

        return outputs
        # logits = self.qa_outputs(with_gcn)
        #
        #
        #
        # # logits = self.qa_outputs(sequence_output)
        # start_logits, end_logits = logits.split(1, dim=-1)
        # start_logits = start_logits.squeeze(-1)
        # end_logits = end_logits.squeeze(-1)
        #
        # outputs = (start_logits, end_logits,) + outputs[2:]
        # if start_positions is not None and end_positions is not None:
        #     # If we are on multi-GPU, split add a dimension
        #     if len(start_positions.size()) > 1:
        #         start_positions = start_positions.squeeze(-1)
        #     if len(end_positions.size()) > 1:
        #         end_positions = end_positions.squeeze(-1)
        #     # sometimes the start/end positions are outside our model inputs, we ignore these terms
        #     ignored_index = start_logits.size(1)
        #     start_positions.clamp_(0, ignored_index)
        #     end_positions.clamp_(0, ignored_index)
        #
        #     loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduce=False)
        #
        #     start_losses = [(loss_fct(start_logits, _start_positions) * _span_mask) \
        #                     for (_start_positions, _span_mask) \
        #                     in zip(torch.unbind(start_positions, dim=1), torch.unbind(answer_masks, dim=1))]
        #     end_losses = [(loss_fct(end_logits, _end_positions) * _span_mask) \
        #                   for (_end_positions, _span_mask) \
        #                   in zip(torch.unbind(end_positions, dim=1), torch.unbind(answer_masks, dim=1))]
        #
        #     total_loss = sum(start_losses + end_losses)
        #     total_loss = torch.mean(total_loss) / 2
        #
        #     outputs = (total_loss,) + outputs
        #
        # return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


class RobertaForCopy(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForCopy, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)

        self.coef = nn.Parameter(torch.ones(config.hidden_size) / config.hidden_size)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head.decoder, self.roberta.embeddings.word_embeddings)

    def get_range_vector(self, size, device):
        """
        Returns a range vector with the desired size, starting at 0. The CUDA implementation
        is meant to avoid copy data from CPU to GPU.
        """
        if device > -1:
            return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
        else:
            return torch.arange(0, size, dtype=torch.long)

    def get_device_of(self, tensor):
        """
        Returns the device of the tensor.
        """
        if not tensor.is_cuda:
            return -1
        else:
            return tensor.get_device()

    def flatten_and_batch_shift_indices(self, indices, sequence_length):
        # Shape: (batch_size)
        if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
            assert (False)
            # raise ConfigurationError(
            #     f"All elements in indices should be in range (0, {sequence_length - 1})"
            # )
        offsets = self.get_range_vector(indices.size(0), self.get_device_of(indices)) * sequence_length
        for _ in range(len(indices.size()) - 1):
            offsets = offsets.unsqueeze(1)

        # Shape: (batch_size, d_1, ..., d_n)
        offset_indices = indices + offsets

        # Shape: (batch_size * d_1 * ... * d_n)
        offset_indices = offset_indices.view(-1)
        return offset_indices

    def batched_index_select(self,
                             target,
                             indices,
                             flattened_indices=None,
                             ):
        if flattened_indices is None:
            # Shape: (batch_size * d_1 * ... * d_n)
            flattened_indices = self.flatten_and_batch_shift_indices(indices, target.size(1))

        # Shape: (batch_size * sequence_length, embedding_size)
        flattened_target = target.view(-1, target.size(-1))

        # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
        flattened_selected = flattened_target.index_select(0, flattened_indices)
        selected_shape = list(indices.size()) + [target.size(-1)]
        # Shape: (batch_size, d_1, ..., d_n, embedding_size)
        selected_targets = flattened_selected.view(*selected_shape)
        return selected_targets

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                masked_lm_labels=None, span_start=None, label_masks=None):

        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        loss_fct = CrossEntropyLoss(ignore_index=-1)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))

        bsz, seq_len = input_ids.size()
        K = span_start.size(1)
        valid_span_mask = (span_start > 0).to(sequence_output.dtype)
        span_start_emb = self.batched_index_select(sequence_output, span_start)
        span_start_emb = span_start_emb * self.coef.unsqueeze(0).unsqueeze(0)  # bsz, K, dim

        scores = torch.matmul(span_start_emb, sequence_output.transpose(1, 2))  # bsz, K, seq_len

        label_masks = label_masks.to(scores.dtype)

        log_norm = torch.logsumexp(scores, dim=-1)
        log_correct = torch.logsumexp(scores - 10000 * (1 - label_masks), dim=-1)

        copy_loss = - torch.sum((log_correct - log_norm) * valid_span_mask) / (torch.sum(valid_span_mask) + 1e-12)

        return masked_lm_loss, copy_loss


class RobertaForQuestionAnsweringForQUOREF(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForQuestionAnsweringForQUOREF, self).__init__(config)
        self.num_labels = config.num_labels
        self.num_answers = config.num_answers

        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, self.num_labels)
        self.qa_classifier = nn.Linear(config.hidden_size, self.num_answers)

        self.point_outputs = nn.Linear(config.hidden_size, 1)

        head_num = config.num_attention_heads // 4

        self.coref_config = BertConfig(num_hidden_layers=1, num_attention_heads=head_num,
                                       hidden_size=config.hidden_size, intermediate_size=256 * head_num)

        self.coref_layer = BertLayer(self.coref_config)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None, answer_masks=None, answer_nums=None):

        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)

        sequence_output_0 = outputs[0]

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        sequence_output = self.coref_layer(sequence_output_0, extended_attention_mask)[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        switch_logits = self.qa_classifier(sequence_output[:, 0, :])

        outputs = (start_logits, end_logits, switch_logits) + outputs[2:]

        if start_positions is not None and end_positions is not None:

            if len(answer_nums.size()) > 1:
                answer_nums = answer_nums.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduce=False)

            start_losses = [(loss_fct(start_logits, _start_positions) * _span_mask) \
                            for (_start_positions, _span_mask) \
                            in zip(torch.unbind(start_positions, dim=1), torch.unbind(answer_masks, dim=1))]
            end_losses = [(loss_fct(end_logits, _end_positions) * _span_mask) \
                          for (_end_positions, _span_mask) \
                          in zip(torch.unbind(end_positions, dim=1), torch.unbind(answer_masks, dim=1))]

            s_e_loss = sum(start_losses + end_losses)
            switch_loss = loss_fct(switch_logits, answer_nums)

            total_loss = torch.mean(s_e_loss + switch_loss)

            outputs = (total_loss,) + outputs

        return outputs


class RobertaForQuestionAnsweringForMRQA(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForQuestionAnsweringForMRQA, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            answer_masks=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduce=False)

            start_losses = [(loss_fct(start_logits, _start_positions) * _span_mask) \
                            for (_start_positions, _span_mask) \
                            in zip(torch.unbind(start_positions, dim=1), torch.unbind(answer_masks, dim=1))]
            end_losses = [(loss_fct(end_logits, _end_positions) * _span_mask) \
                          for (_end_positions, _span_mask) \
                          in zip(torch.unbind(end_positions, dim=1), torch.unbind(answer_masks, dim=1))]

            total_loss = sum(start_losses + end_losses)
            total_loss = torch.mean(total_loss) / 2

            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


class RobertaForSequenceEncoder(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForTokenClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForSequenceEncoder, self).__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=None,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)

        sequence_output = self.dropout(outputs[0])
        pooled_output = self.dropout(outputs[1])

        return sequence_output, pooled_output

# import dgl
# import dgl.function as fn
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import List, Union
# import torch
# from dgl import DGLGraph
#
# gcn_msg = fn.copy_src(src="h", out="m")
# gcn_reduce = fn.sum(msg="m", out="h")
#
#
# class NodeApplyModule(nn.Module):
#     def __init__(self, in_feats, out_feats, activation):
#         super(NodeApplyModule, self).__init__()
#         self.linear = nn.Linear(in_feats, out_feats)
#         self.activation = activation
#
#     def forward(self, node):
#         h = self.linear(node.data["h"])
#         if self.activation is not None:
#             h = self.activation(h)
#         return {"h": h}
#
#
# class GCN(nn.Module):
#     def __init__(self, in_feats, out_feats, activation):
#         super(GCN, self).__init__()
#         self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)
#
#     def forward(self, g, feature):
#         g.ndata["h"] = feature
#         g.update_all(gcn_msg, gcn_reduce)
#         g.apply_nodes(func=self.apply_mod)
#         return g.ndata.pop("h")
#
#
# class GCNNet(nn.Module):
#     def __init__(self, hdim, nlayers=2, dropout_prob=0.1):
#         super(GCNNet, self).__init__()
#         # self.gcns = nn.ModuleList([GCN(hdim, hdim, F.relu) for i in range(nlayers)])
#         feedfoward_input_dim, feedforward_hidden_dim, hidden_dim = hdim, hdim, hdim
#         self._gcn_layers = [GCN(hdim, hdim, F.relu) for _ in range(nlayers)]
#         self.nlayers = nlayers
#         self.linear = nn.Linear(hdim, hdim)
#         # self._feedfoward_layers = [FeedForward(feedfoward_input_dim,
#         #                              activations=[F.relu,
#         #                                           nn.Linear(hdim, hdim, bias=True)],
#         #                              hidden_dims=[feedforward_hidden_dim, hidden_dim],
#         #                              num_layers=2,
#         #                              dropout=[dropout_prob, dropout_prob]) for _ in range(nlayers)]
#         self._layer_norm_layers = [nn.LayerNorm(hdim) for _ in range(nlayers)]
#         self._feed_forward_layer_norm_layers = [nn.LayerNorm(hdim) for _ in range(nlayers)]
#
#         self.dropout = nn.Dropout(dropout_prob)
#         self._input_dim = hdim
#         self._output_dim = hdim
#
#     def forward(self, g, features):
#         output = features
#         for i in range(self.nlayers):
#             gcn = self._gcn_layers[i]
#             # feedforward = self._feedfoward_layers[i]
#             feedforward_layer_norm = self._feed_forward_layer_norm_layers[i]
#             layer_norm = self._layer_norm_layers[i]
#             cached_input = output
#             feedforward_output = F.relu(self.linear(output))
#             feedforward_output = self.dropout(feedforward_output)
#             if feedforward_output.size() == cached_input.size():
#                 feedforward_output = feedforward_layer_norm(feedforward_output + cached_input)
#             # shape (batch_size, sequence_length, hidden_dim)
#             attention_output = gcn(g, feedforward_output)
#             output = layer_norm(self.dropout(attention_output) + feedforward_output)
#         return output
#
#
# class GCN_layers(nn.Module):
#
#     def __init__(self, hdim,
#                  nlayers=2
#                  ):
#         super(GCN_layers, self).__init__()
#         self.GCNNet = GCNNet(hdim, nlayers, 0.1)
#
#     def transform_sent_rep(self, sent_rep, sent_mask, meta_field, key):
#         init_graphs = self.convert_sent_tensors_to_graphs(sent_rep, sent_mask, meta_field, key)
#         unpadated_graphs = []
#         for g in init_graphs:
#             updated_graph = self.forward(g)
#             unpadated_graphs.append(updated_graph)
#         recovered_sent = torch.stack(unpadated_graphs, dim=0)
#         assert recovered_sent.shape == sent_rep.shape
#         return recovered_sent
#
#     def forward(self, g):
#         h = g.ndata['h']
#         out = self.GCNNet.forward(g, features=h)
#         # return g, g.ndata['h'], hg  # g is the raw graph, h is the node rep, and hg is the mean of all h
#         print(out)
#         return out
#
#     def get_input_dim(self) -> int:
#         return self.input_dim
#
#     def get_output_dim(self) -> int:
#         return self._output_dim
#
#     def is_bidirectional(self):
#         return False


import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from functools import partial
gcn_msg=fn.copy_src(src="h",out="m")
gcn_reduce=fn.sum(msg="m",out="h")

class RGCNLayer(nn.Module):
    def __init__(self, feat_size, num_rels, activation=None, gated=True):

        super(RGCNLayer, self).__init__()
        self.feat_size = feat_size
        self.num_rels = num_rels
        self.activation = activation
        self.gated = gated

        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.feat_size, self.feat_size))
        # init trainable parameters
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        if self.gated:
            self.gate_weight = nn.Parameter(torch.Tensor(self.num_rels, self.feat_size, 1))
            nn.init.xavier_uniform_(self.gate_weight, gain=nn.init.calculate_gain('sigmoid'))

    def forward(self, g):

        weight = self.weight
        gate_weight = self.gate_weight

        def message_func(edges):
            w = weight[edges.data['rel_type']]
            # print(edges.src["h"].shape)
            # print(w.shape)
            msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
            msg = msg * edges.data['norm']

            if self.gated:
                gate_w = gate_weight[edges.data['rel_type']]
                gate = torch.bmm(edges.src['h'].unsqueeze(1), gate_w).squeeze().reshape(-1, 1)
                gate = torch.sigmoid(gate)
                msg = msg * gate
            return {'msg': msg}

        def apply_func(nodes):
            h = nodes.data['h']
            h = self.activation(h)
            return {'h': h}

        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)


class RGCNModel(nn.Module):
    def __init__(self, h_dim, num_rels, num_hidden_layers=1, gated=True):
        super(RGCNModel, self).__init__()

        # self.h_dim = h_dim
        self.h_dim = 32
        self.num_rels = num_rels
        self.num_hidden_layers = num_hidden_layers
        self.gated = gated

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        for _ in range(self.num_hidden_layers):
            rgcn_layer = RGCNLayer(self.h_dim, self.num_rels, activation=F.relu, gated=self.gated)
            self.layers.append(rgcn_layer)

    def forward(self, g):
        for layer in self.layers:
            layer(g)

        rst_hidden = []
        for sub_g in dgl.unbatch(g):
            rst_hidden.append(sub_g.ndata['h'])
        return rst_hidden