import argparse
import json
import logging
import math
import collections
from io import open
import numpy as np
import pickle

from transformers_coref import RobertaTokenizer

from transformers_coref.tokenization_bert import BasicTokenizer, whitespace_tokenize

# Required by XLNet evaluation method to compute optimal threshold (see write_predictions_extended() method)
import random
import re
import string

import spacy
nlp = spacy.load('en_core_web_sm')

logger = logging.getLogger(__name__)


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 # doc_span_start,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_index,
                 p_mask,
                 paragraph_len,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 answer_mask=None,
                 answer_num=None
                 ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.answer_mask = answer_mask
        self.answer_num = answer_num


def read_squad_examples(input_file, is_training, version_2_with_negative):
  """Read a SQuAD json file into a list of SquadExample."""
  with open(input_file, "r", encoding='utf-8') as reader:
    input_data = json.load(reader)["data"]

  input_data = input_data

  def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
      return True
    return False

  examples = []
  for entry in input_data:
    for paragraph in entry["paragraphs"]:
      paragraph_text = paragraph["context"]

      doc = nlp(paragraph_text)
      doc_tokens = []
      # print(nltk.pos_tag(nltk.word_tokenize(s)))
      char_to_word_offset = []
      tokenindex = []
      start = 0
      for token in doc:
        doc_tokens.append(token.text)
        index = paragraph_text[start:].find(token.text)
        tokenindex.append(start + index)
        start = start + index + len(token)
      for i in range(len(tokenindex)):
        length = 0
        if i == len(tokenindex) - 1:
          length = len(paragraph_text) - tokenindex[i]
        else:
          length = tokenindex[i + 1] - tokenindex[i]
        for j in range(length):
          char_to_word_offset.append(i)
      '''doc_tokens = []
      char_to_word_offset = []
      prev_is_whitespace = True
      for c in paragraph_text:
          if is_whitespace(c):
              prev_is_whitespace = True
          else:
              if prev_is_whitespace:
                  doc_tokens.append(c)
              else:
                  doc_tokens[-1] += c
              prev_is_whitespace = False
          char_to_word_offset.append(len(doc_tokens) - 1)'''

      for qa in paragraph["qas"]:
        qas_id = qa["id"]

        question_text = qa["question"]
        start_positions = []
        end_positions = []
        orig_answer_texts = []
        is_impossible = False

        if is_training:  # for debug
          if version_2_with_negative:
            is_impossible = qa.get("is_impossible", False)

          if not is_impossible:
            flag = True

            for answer in qa["answers"]:
              orig_answer_text = answer["text"]
              answer_offset = answer["answer_start"]
              answer_length = len(orig_answer_text)
              start_position = char_to_word_offset[answer_offset]
              end_position = char_to_word_offset[answer_offset + answer_length - 1]
              # Only add answers where the text can be exactly recovered from the
              # document. If this CAN'T happen it's likely due to weird Unicode
              # stuff so we will just skip the example.
              #
              # Note that this means for training mode, every example is NOT
              # guaranteed to be preserved.
              actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
              cleaned_answer_text = " ".join(
                whitespace_tokenize(orig_answer_text))
              # if cleaned_answer_text.endswith(","):
              #   cleaned_answer_text = cleaned_answer_text[:-1].strip()
              if actual_text.find(cleaned_answer_text) == -1:
                print(qas_id)
                logger.warning("Could not find answer: '%s' vs. '%s'",
                               actual_text, cleaned_answer_text)

                flag = False
                break
              start_positions.append(start_position)
              end_positions.append(end_position)
              orig_answer_texts.append(orig_answer_text)

            if not flag and is_training:
              continue

          # else:
          #     start_position = -1
          #     end_position = -1
          #     orig_answer_text = ""

        example = SquadExample(
          qas_id=qas_id,
          question_text=question_text,
          doc_tokens=doc_tokens,
          orig_answer_text=orig_answer_texts,
          start_position=start_positions,
          end_position=end_positions,
          is_impossible=is_impossible
        )
        examples.append(example)

  return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True, max_n_answers=1):
  """Loads a data file into a list of `InputBatch`s."""

  unique_id = 1000000000

  features = []
  for (example_index, example) in enumerate(examples):

    # print(example_index, example.question_text)
    # if example.question_text is None or len(example.question_text) < 1:
    #     continue
    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:
      query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
      orig_to_tok_index.append(len(all_doc_tokens))
      sub_tokens = tokenizer.tokenize(token)
      for sub_token in sub_tokens:
        tok_to_orig_index.append(i)
        all_doc_tokens.append(sub_token)

    tok_start_positions = []
    tok_end_positions = []
    if is_training and example.is_impossible:
      tok_start_positions.append(-1)
      tok_end_positions.append(-1)

    if is_training and not example.is_impossible:

      for orig_answer_text, start_position, end_position in zip(example.orig_answer_text, example.start_position,
                                                                example.end_position):
        tok_start_position = orig_to_tok_index[start_position]
        if end_position < len(example.doc_tokens) - 1:
          tok_end_position = orig_to_tok_index[end_position + 1] - 1
        else:
          tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
          all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
          orig_answer_text)

        tok_start_positions.append(tok_start_position)
        tok_end_positions.append(tok_end_position)

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
      "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
      length = len(all_doc_tokens) - start_offset
      if length > max_tokens_for_doc:
        length = max_tokens_for_doc
      doc_spans.append(_DocSpan(start=start_offset, length=length))
      if start_offset + length == len(all_doc_tokens):
        break
      start_offset += min(length, doc_stride)
      while start_offset < len(all_doc_tokens) and all_doc_tokens[start_offset - 1] != ".":
        start_offset += 1

    for (doc_span_index, doc_span) in enumerate(doc_spans):
      tokens = []
      token_to_orig_map = {}
      token_is_max_context = {}
      segment_ids = []

      # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
      # Original TF implem also keep the classification token (set to 0) (not sure why...)
      p_mask = []

      # CLS token at the beginning
      if not cls_token_at_end:
        tokens.append(cls_token)
        segment_ids.append(cls_token_segment_id)
        p_mask.append(0)
        cls_index = 0

      # Query
      for token in query_tokens:
        tokens.append(token)
        segment_ids.append(sequence_a_segment_id)
        p_mask.append(1)

      # SEP token
      tokens.append(sep_token)
      segment_ids.append(sequence_a_segment_id)
      p_mask.append(1)

      # Paragraph
      for i in range(doc_span.length):
        split_token_index = doc_span.start + i
        token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

        is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                               split_token_index)
        token_is_max_context[len(tokens)] = is_max_context
        tokens.append(all_doc_tokens[split_token_index])
        segment_ids.append(sequence_b_segment_id)
        p_mask.append(0)
      paragraph_len = doc_span.length

      # SEP token
      tokens.append(sep_token)
      segment_ids.append(sequence_b_segment_id)
      p_mask.append(1)

      # CLS token at the end
      if cls_token_at_end:
        tokens.append(cls_token)
        segment_ids.append(cls_token_segment_id)
        p_mask.append(0)
        cls_index = len(tokens) - 1  # Index of classification token

      input_ids = tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

      # Zero-pad up to the sequence length.
      while len(input_ids) < max_seq_length:
        input_ids.append(pad_token)
        input_mask.append(0 if mask_padding_with_zero else 1)
        segment_ids.append(pad_token_segment_id)
        p_mask.append(1)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length

      span_is_impossible = example.is_impossible
      # start_position = None
      # end_position = None

      start_positions = []
      end_positions = []

      doc_start = doc_span.start
      doc_end = doc_span.start + doc_span.length - 1

      if is_training and not span_is_impossible:
        # For training, if our document chunk does not contain an annotation
        # we throw it out, since there is nothing to predict.
        doc_start = doc_span.start
        doc_end = doc_span.start + doc_span.length - 1

        for tok_start_position, tok_end_position in zip(tok_start_positions, tok_end_positions):
          if not (tok_start_position >= doc_start and
                  tok_end_position <= doc_end):
            continue

          doc_offset = len(query_tokens) + 2
          start_position = tok_start_position - doc_start + doc_offset
          end_position = tok_end_position - doc_start + doc_offset

          start_positions.append(start_position)
          end_positions.append(end_position)

      if is_training and len(start_positions) == 0:  # and span_is_impossible:
        start_positions = [cls_index]
        end_positions = [cls_index]
        span_is_impossible = True

      # if example_index < 10:
      #     logger.info("*** Example ***")
      #     logger.info("example_index: %s" % (example_index))
      #     logger.info("tokens: %s" % " ".join(tokens))

      #     if is_training and not span_is_impossible and len(example.orig_answer_text)>1:
      #         print (example.orig_answer_text)
      #         for start_position, end_position in zip(start_positions, end_positions):
      #             answer_text = " ".join(tokens[start_position:(end_position + 1)])
      #             # logger.info("start_position: %d" % (start_position))
      #             # logger.info("end_position: %d" % (end_position))
      #             logger.info("answer: %s" % (answer_text))

      if len(start_positions) > max_n_answers:
        idxs = np.random.choice(len(start_positions), max_n_answers, replace=False)
        st = []
        en = []
        for idx in idxs:
          st.append(start_positions[idx])
          en.append(end_positions[idx])
        start_positions = st
        end_positions = en

      answer_num = len(start_positions) if not span_is_impossible else 0

      answer_mask = [1 for _ in range(len(start_positions))]
      for _ in range(max_n_answers - len(start_positions)):
        start_positions.append(0)
        end_positions.append(0)
        answer_mask.append(0)

      features.append(
        InputFeatures(
          unique_id=unique_id,
          example_index=example_index,
          doc_span_index=doc_span_index,
          tokens=tokens,
          token_to_orig_map=token_to_orig_map,
          token_is_max_context=token_is_max_context,
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          cls_index=cls_index,
          p_mask=p_mask,
          paragraph_len=paragraph_len,
          start_position=start_positions,
          end_position=end_positions,
          is_impossible=span_is_impossible,
          answer_mask=answer_mask,
          answer_num=answer_num
        ))
      unique_id += 1

  return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index



parser = argparse.ArgumentParser()

## Other parameters
parser.add_argument("--max_n_answers", default=2, type=int, help="")

parser.add_argument("--config_name", default="", type=str,
                    help="Pretrained config name or path if not the same as model_name")
parser.add_argument("--tokenizer_name", default="", type=str,
                    help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--cache_dir", default="", type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")

parser.add_argument('--version_2_with_negative', action='store_true',
                    help='If true, the SQuAD examples contain some that do not have an answer.')
parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                    help="If null_score - best_non_null is greater than the threshold predict null.")

parser.add_argument("--max_seq_length", default=512, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                         "longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--doc_stride", default=128, type=int,
                    help="When splitting up a long document into chunks, how much stride to take between chunks.")
parser.add_argument("--max_query_length", default=64, type=int,
                    help="The maximum number of tokens for the question. Questions longer than this will "
                         "be truncated to this length.")
parser.add_argument("--do_train", action='store_true',
                    help="Whether to run training.")
parser.add_argument("--do_eval", action='store_true',
                    help="Whether to run eval on the dev set.")
parser.add_argument("--evaluate_during_training", action='store_true',
                    help="Rul evaluation during training at each logging step.")
parser.add_argument("--do_lower_case", action='store_true',
                    help="Set this flag if you are using an uncased model.")

parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--learning_rate", default=5e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=3.0, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_steps", default=-1, type=int,
                    help="Linear warmup over warmup_steps.")
parser.add_argument("--n_best_size", default=20, type=int,
                    help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
parser.add_argument("--max_answer_length", default=30, type=int,
                    help="The maximum length of an answer that can be generated. This is needed because the start "
                         "and end predictions are not conditioned on one another.")
parser.add_argument("--verbose_logging", action='store_true',
                    help="If true, all of the warnings related to data processing will be printed. "
                         "A number of warnings are expected for a normal SQuAD evaluation.")

parser.add_argument('--logging_steps', type=int, default=50,
                    help="Log every X updates steps.")
parser.add_argument('--save_steps', type=int, default=884,
                    help="Save checkpoint every X updates steps.")
parser.add_argument("--eval_all_checkpoints", action='store_true',
                    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
parser.add_argument("--no_cuda", action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument('--overwrite_output_dir', action='store_true',
                    help="Overwrite the content of the output directory")
parser.add_argument('--overwrite_cache', action='store_true',
                    help="Overwrite the cached training and evaluation sets")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")

parser.add_argument("--local_rank", type=int, default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--fp16', action='store_true',
                    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
parser.add_argument('--fp16_opt_level', type=str, default='O1',
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

args = parser.parse_args()
file_name = "quoref/traintest.json"
text = "0 What is the first name of the person who doubted it would turn out to be a highly explosive eruption like those that can occur in subduction-zone volcanoes?"
tokenizer = RobertaTokenizer.from_pretrained("coref-roberta-large")
tokens = tokenizer.tokenize(text)
examples = read_squad_examples(file_name, True, False)
features = convert_examples_to_features(examples=examples,
                                        tokenizer=tokenizer,
                                        max_seq_length=args.max_seq_length,
                                        doc_stride=args.doc_stride,
                                        max_query_length=args.max_query_length,
                                        is_training=True,
                                        cls_token='<s>',
                                        sep_token='</s>',
                                        max_n_answers=args.max_n_answers)
