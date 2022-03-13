""" Load SQuAD dataset. """

from __future__ import absolute_import, division, print_function

import json
import logging
import math
import collections
from io import open
import numpy as np
import pickle
from transformers_coref.tokenization_bert import BasicTokenizer, whitespace_tokenize

# Required by XLNet evaluation method to compute optimal threshold (see write_predictions_extended() method)

import random
import re
import string

import spacy
import neuralcoref
nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)

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
                 doc_coref,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.doc_coref = doc_coref
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
                 doc_coref,
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
        self.doc_coref = doc_coref
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
            coref_clusters = doc._.coref_clusters
            doc_tokens = []
            doc_coref = []  # token的coref信息
            char_to_word_offset = []
            tokenindex = []
            start = 0
            for token in doc:
                doc_tokens.append(token.text)
                doc_coref.append(0)
                index = paragraph_text[start:].find(token.text)
                tokenindex.append(start + index)
                start = start + index + len(token)
            # 标记coref关系,+1是不是太多了？？
            for (cluster_index, cluster) in enumerate(coref_clusters):
                for item in cluster.mentions:
                    for span_index in range(item.start, item.end):
                        doc_coref[span_index] = cluster_index + 1

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
                    doc_coref=doc_coref,
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

        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        all_doc_coref = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            # print('token:', token)
            # print('subtokens:', sub_tokens)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
                all_doc_coref.append(example.doc_coref[i])

        tok_start_positions = []
        tok_end_positions = []
        if is_training and example.is_impossible:
            tok_start_positions.append(-1)
            tok_end_positions.append(-1)
        
        if is_training and not example.is_impossible:
                
            for orig_answer_text, start_position, end_position in zip(example.orig_answer_text, example.start_position, example.end_position):
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
            while start_offset < len(all_doc_tokens) and all_doc_tokens[start_offset - 1]!=".":
                start_offset += 1

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            coref = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = []

            # CLS token at the beginning
            if not cls_token_at_end:
                tokens.append(cls_token)
                coref.append(0)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = 0

            # Query
            for token in query_tokens:
                tokens.append(token)
                coref.append(0)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

            # SEP token
            tokens.append(sep_token)
            coref.append(0)
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
                coref.append(all_doc_coref[split_token_index])
                segment_ids.append(sequence_b_segment_id)
                p_mask.append(0)
            paragraph_len = doc_span.length

            # SEP token
            tokens.append(sep_token)
            coref.append(0)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)

            # CLS token at the end
            if cls_token_at_end:
                tokens.append(cls_token)
                coref.append(0)
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
                coref.append(0)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)

            assert len(input_ids) == max_seq_length
            assert len(coref) == max_seq_length
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
                    
            if is_training and len(start_positions)==0:#and span_is_impossible:
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
                    doc_coref=coref,
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

RawResult_multi = collections.namedtuple("RawResult_multi",
                                   ["unique_id", "start_logits", "end_logits", "answer_num"])


def write_predictions_multi(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    # logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    id2fakeanswer = {}
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        fake_answer = []

        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        answer_num = 1
        for (feature_index, feature) in enumerate(features):


            result = unique_id_to_result[feature.unique_id]

            # for debug
            answer_nums = result.answer_num
            if answer_nums[2]>answer_nums[1]:
                answer_num = 2

            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)

            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
                    if start_index>0:
                        fake_answer.append(_PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        
        fake_answer = sorted(
            fake_answer,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        nbest = []

        for pred in fake_answer:
            feature = features[pred.feature_index]
            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]

            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()

            # fix Father 's day and 85 % percent bugs
            new_tok_text = ""
            tokens = tok_text.split()
            i = 0
            while i < len(tokens):
                if tokens[i] == "\'s" or tokens[i] == "%":
                    new_tok_text = new_tok_text + tokens[i]
                else:
                    new_tok_text = new_tok_text + " " + tokens[i]
                i = i + 1

            tok_text = new_tok_text.strip()

            orig_text = ""
            i = 0
            while i < len(orig_tokens):
                if orig_tokens[i] == "\'s" or orig_tokens[i] == "%":
                    orig_text = orig_text + orig_tokens[i]
                else:
                    orig_text = orig_text + " " + orig_tokens[i]
                i = i + 1
            orig_text = orig_text.strip()

            final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
            # def normalize_answer(s):
            #     def remove_articles(text):
            #         regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            #         return re.sub(regex, ' ', text)
            #     def white_space_fix(text):
            #         return ' '.join(text.split())
            #     def remove_punc(text):
            #         exclude = set(string.punctuation)
            #         return ''.join(ch for ch in text if ch not in exclude)
            #     def lower(text):
            #         return text.lower()
            #     return white_space_fix(remove_articles(remove_punc(lower(s))))
            def normalize_answer(s):
                def remove_articles(text):
                    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
                    return re.sub(regex, ' ', text)
                def white_space_fix(text):
                    return ' '.join(text.split())
                def remove_punc(text):
                    exclude = set(string.punctuation)
                    # if text[0] in exclude:
                    #     text = text[1:]
                    # if text[-1] in exclude:
                    #     text = text[:-1]
                    # return text
                    return ''.join(ch for ch in text if ch not in exclude)
                def lower(text):
                    return text.lower()

                def removespace(text):
                    s = re.sub(r"([a-zA-Z]) ('s\b)", r"\1\2", text)
                    s = re.sub(r"(\d) %", r"\1%", s)
                    return s

                results = white_space_fix(remove_articles(remove_punc(lower(s))))
                return results

            # def normalize_answer(s):
            #     def remove_articles(text):
            #         regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            #         return re.sub(regex, ' ', text)
            #
            #     def white_space_fix(text):
            #         return ' '.join(text.split())
            #
            #     def remove_punc(text):
            #         exclude = set(string.punctuation)
            #         return ''.join(ch for ch in text if ch not in exclude)
            #
            #     def lower(text):
            #         return text.lower()
            #
            #     def removespace(text):
            #         s = re.sub(r"([a-zA-Z]) ('s\b)", r"\1\2", text)
            #         s = re.sub(r"(\d) %", r"\1%", s)
            #         return s
            #
            #     return removespace(white_space_fix(remove_articles(remove_punc(lower(s)))))

            def get_tokens(s):
                if not s: return []
                return normalize_answer(s).split()

            def compute_f1(a_gold, a_pred):
                gold_toks = get_tokens(a_gold)
                pred_toks = get_tokens(a_pred)
                common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
                num_same = sum(common.values())
                if len(gold_toks) == 0 or len(pred_toks) == 0:
                    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
                    return int(gold_toks == pred_toks)
                if num_same == 0:
                    return 0
                precision = 1.0 * num_same / len(pred_toks)
                recall = 1.0 * num_same / len(gold_toks)
                f1 = (2 * precision * recall) / (precision + recall)
                return f1



            overlap = False   # 有空改成算f1 的
            for item in nbest:
                # if (orig_doc_start-5 <= item["orig_doc_start"] and item["orig_doc_start"] <= orig_doc_end+5) or (orig_doc_start-5 <= item["orig_doc_end"] and item["orig_doc_end"] <= orig_doc_end+5):
                #     overlap = True
                #     break
                if item['text'].find(final_text)!=-1 or final_text.find(item['text'])!=-1:
                    overlap = True
                    break
                if compute_f1(item['text'], final_text)>0:
                    overlap = True
                    break

            if overlap:
                continue

            nbest.append({"orig_doc_start":orig_doc_start, "orig_doc_end": orig_doc_end, "text": final_text})
            if len(nbest)==2:
                break     
        id2fakeanswer[feature.example_index] = nbest


        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit", "orig_doc_start", "orig_doc_end"])

        answer_num = max(answer_num, 1)

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= answer_num:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]


                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                overlap = False   # 有空改成算f1 的
                for item in nbest:
                    if (orig_doc_start <= item.orig_doc_start and item.orig_doc_start <= orig_doc_end) or (orig_doc_start <= item.orig_doc_end and item.orig_doc_end <= orig_doc_end):
                        overlap = True
                    elif item.text.find(final_text)!=-1 or final_text.find(item.text)!=-1:
                        overlap = True
                if overlap:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True
                orig_doc_start = orig_doc_end = -1


            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    orig_doc_start = orig_doc_start,
                    orig_doc_end = orig_doc_end,
                    ))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit,
                        orig_doc_start=-1, orig_doc_end=-1))

            # # In very rare edge cases we could only have single null prediction.
            # # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0,
                             _NbestPrediction(text="empty", start_logit=0, end_logit=0, orig_doc_start=-1, orig_doc_end=-1))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0, end_logit=0, orig_doc_start=-1, orig_doc_end=-1))

        assert len(nbest) >= 1
        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            # if example.qas_id!="c090e6763e32738d9db4615f4f9065d44525c809":
            #     continue
            ans = []
            for i in range(min(answer_num, len(nbest_json))):
                ans.append(nbest_json[i]["text"])
            
            all_predictions[example.qas_id] = ans#nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            score_diff = 0
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    # with open(output_nbest_file, "w") as writer:
    #     writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(id2fakeanswer, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs
