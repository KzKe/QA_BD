import transformers
from transformers import BertTokenizer

from config import config

import os 
import csv 
import json
import random 
import logging 
import argparse 
import numpy as np 
from tqdm import tqdm 

import string 
import re


import torch 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


tokenizer = config.TOKENIZER

class InputExample(object):
    def __init__(self, 
        qas_id, 
        question_text, 
        context_text, 
        doc_tokens = None, 
        answer_text=None, 
        start_position=None, 
        end_position=None, 
        # is_impossible=None, 
        # ner_cate=None
        ):

        """
        Desc:
            is_impossible: bool, [True, False]
        """

        self.qas_id = qas_id 
        self.question_text = question_text 
        self.context_text = context_text 
        self.answer_text = answer_text 
        self.start_position = start_position 
        self.end_position = end_position 



def read_question_answer_examples(input_file, is_training=True, with_negative=True):
    """
    Desc:
        read question-answering data
    """

    with open(input_file, "r") as f:
      input_data = json.load(f)['data']

    examples = []
    for entry in tqdm(input_data):
        title = entry["title"]
        for paragraph in entry["paragraphs"]:
            context_text = paragraph["context"]
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]

                answer = qa["answers"][0]
                answer_text = answer["text"]
                assert answer_text != None 
                start_position = answer["answer_start"]

                example = InputExample(qas_id=qas_id, 
                                      question_text=question_text, 
                                      context_text=context_text,
                                      start_position=start_position, 
                                      end_position=None,
                                      answer_text = answer_text
                                      # is_impossible=is_impossible, 
                                      # ner_cate=ner_cate
                                       )
                          
                examples.append(example)
    print(len(examples))
    return examples







class InputFeatures(object):
    """
    Desc:
        a single set of features of data 
    Args:
        start_pos: start position is a list of symbol 
        end_pos: end position is a list of symbol 
    """
    def __init__(self, 
                unique_id, 
                input_ids, 
                input_mask, 
                token_type_ids, 
                start_position, 
                end_position, 
                doc_text,
                orig_answer_text,
                query_text
                ):

        self.unique_id = unique_id 
        self.input_mask = input_mask
        self.input_ids = input_ids 
        self.token_type_ids = token_type_ids 
        self.start_position = start_position 
        self.end_position = end_position 
        self.doc_text = doc_text,
        self.orig_answer_text=orig_answer_text,
        self.query_text=query_text
 


def add_space(text):
  # add space for .com
  text = text.replace('com', 'com ')
  # add space for numbers 
  nums = re.findall(r'\d+', text)
  for num in nums:
    pos = text.find(num)
    if pos-1>=0 and text[pos-1].isnumeric():
      text = text
    else:
      text = text[:pos] +' ' + text[pos:]
  return text



def process_data(doc_text, query_text, orig_answer_text, tokenizer, max_len=512, doc_max_len=400, query_max_len=100,unique_id=0):

    new_text = doc_text
    doc_text = add_space(doc_text.lower())
    doc_tokens = tokenizer.tokenize(doc_text)
    answer_token = tokenizer.tokenize(add_space(orig_answer_text.lower()))
    query_token = tokenizer.tokenize(query_text.lower())

    idx0 = None
    idx1 = None
    len_st = len(answer_token)
    for ind in (i for i, e in enumerate(doc_tokens) if e == answer_token[0]):
      if doc_tokens[ind: ind+len_st] == answer_token:
          idx0 = ind
          idx1 = ind + len_st - 1
          break

    doc_encoded = tokenizer.encode_plus(doc_text, add_special_tokens=False, max_length=doc_max_len )
    answer_encoded = tokenizer.encode_plus(orig_answer_text, add_special_tokens=False)
    query_encoded = tokenizer.encode_plus(query_text, add_special_tokens=False, max_len=query_max_len)

    input_ids = [tokenizer.cls_token_id] + query_encoded['input_ids'] + [tokenizer.sep_token_id] + doc_encoded['input_ids'] +  [tokenizer.sep_token_id]
    token_type_ids = [0]+ query_encoded['token_type_ids'] + [0] + doc_encoded['token_type_ids'] + [0]
    mask = [1] * len(token_type_ids)

    # try:
    if idx0 is None or idx1 is None:
      start_position = -1 
      end_position = -1
      print(orig_answer_text)
    else:
      start_position = idx0 + len(query_encoded['input_ids']) + 2
      end_position = idx1 + len(query_encoded['input_ids']) + 2


    padding_length = max_len - len(input_ids)

    if padding_length > 0:
        input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([tokenizer.pad_token_type_id] * padding_length)

    return InputFeatures(
                          unique_id, 
                          input_ids=input_ids, 
                          input_mask=mask, 
                          token_type_ids=token_type_ids,
                          start_position=start_position, 
                          end_position=end_position, 
                          doc_text = doc_text,
                          orig_answer_text=orig_answer_text,
                          query_text=query_text
                        )


# def convert_examples_to_features(examples, tokenizer, max_seq_length=512):
#     features = []
#     for example in examples:
#       feature = process_data(
#           doc_text = example.context_text,
#           query_text = example.question_text,
#           orig_answer_text = example.answer_text,
#           tokenizer = tokenizer,
#           max_len=max_seq_length
#       )
#       features.append(feature)
#     return features 



