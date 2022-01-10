import tensorflow as tf
import numpy as np
import csv
from tqdm import tqdm
import json
from copy import copy
import os

from transformers import BertTokenizer, TFBertForMaskedLM, TFBertModel
from transformers import RobertaTokenizer, TFRobertaForMaskedLM, TFRobertaModel
from transformers import XLNetTokenizer, TFXLNetLMHeadModel
from transformers import AlbertTokenizer, TFAlbertForMaskedLM

import argparse

from tf_modified_bert_encoder import TFModifiedBertEncoder
import constants


def get_model_tokenizer(model_path, filter_layers=None, keep_information=False, filter_threshold=1e-4):
    do_lower_case = ('uncased' in model_path or 'albert' in model_path)
    if model_path.startswith('bert'):
        tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=do_lower_case)
        model = TFBertForMaskedLM.from_pretrained(model_path, output_hidden_states=False, output_attentions=False)
        if filter_layers:
            OSPs_out = np.load(f'../experiments/{model_path}-intercept/osp_bias_{model_path}.npz', allow_pickle=True)
            if keep_information:
                OSPs_in = np.load(f'../experiments/{model_path}-intercept/osp_information_{model_path}.npz', allow_pickle=True)
            else:
                OSPs_in = None
                encoder = model.layers[0].encoder
                modified_encoder = TFModifiedBertEncoder(OSPs_out, filter_layers,
                                                        projection_matrices_in=OSPs_in, filter_threshold=filter_threshold, source=encoder)
                model.layers[0].encoder = modified_encoder
    elif model_path.startswith('roberta'):
        tokenizer = RobertaTokenizer.from_pretrained(model_path, do_lower_case=do_lower_case, add_prefix_space=True)
        model = TFRobertaForMaskedLM.from_pretrained(model_path, output_hidden_states=False, output_attentions=False)
    elif model_path.startswith('xlnet'):
        tokenizer = XLNetTokenizer.from_pretrained(model_path, do_lower_case=do_lower_case)
        model = TFXLNetLMHeadModel.from_pretrained(model_path, output_hidden_states=False, output_attentions=False)
    elif model_path.startswith('albert'):
        tokenizer = AlbertTokenizer.from_pretrained(model_path, do_lower_case=do_lower_case, remove_spaces=True)
        model = TFAlbertForMaskedLM.from_pretrained(model_path, output_hidden_states=False, output_attentions=False)
    else:
        raise ValueError(f"Unknown Transformer name: {model_path}. "
                        f"Please select one of the supported models: {constants.SUPPORTED_MODELS}")

    return model, tokenizer


def get_name(args):
    name = f"mlm_accuracy_{args.model}_{args.evaluation_file}"
    if args.filter_layers:
        fl_str = '_'.join(list(map(str,args.filter_layers)))
        name += f"_f{fl_str}"
    
        if args.filter_threshold != 1e-4:
            name += f"_thr_{str(args.filter_threshold)}"
    
        if args.keep_information:
            name += "_keep-information"
    
    return name


def prediction_test(model, tokenizer, sent):
    sent_correct = 0
    sent_predicted = 0
    
   
    mask_id = tokenizer.convert_tokens_to_ids([tokenizer.mask_token])[0]
    tokens = [tokenizer.cls_token] + tokenizer.tokenize(sent) + [tokenizer.sep_token]
    if len(tokens) > 256:
        print(f"Skipping sent: {sent}\nToo many tokens")
        return 0, 0
    token_ids = []
    for t_id, token in enumerate(tokens):
        mod_tokens = tokens.copy()
        mod_tokens[t_id] = tokenizer.mask_token
        token_ids.append(tf.constant(tokenizer.convert_tokens_to_ids(mod_tokens), dtype=tf.int64))

    token_ids = tf.stack(token_ids)

    outputs = model(token_ids)

    for t_id, (token, logits) in enumerate(zip(tokenizer.convert_tokens_to_ids(tokens), tf.unstack(outputs.logits))):
        if tf.argmax(logits[t_id]) == token:
            sent_correct += 1
        sent_predicted +=1

    #inputs = tokenizer(sent, return_tensors="tf")
    #inputs["labels"] = tokenizer(sent, return_tensors="tf")["input_ids"]
    #for t_id, token in enumerate(inputs):
    #   query_inputs = inputs.copy()
    #    print(query_inputs)
    #    query_inputs['input_ids'][t_id] = mask_id
    #    print(query_inputs)

    #     outputs = model(query_inputs)
    #     logits = outputs.logits
    #     if token == tf.argmax(logits):
    #         sent_correct += 1
    #     sent_predicted += 1

    return sent_correct, sent_predicted
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("data_dir", help="Directory with data")
    parser.add_argument("--evaluation-file", default='en_ewt-ud-test', type=str, help="File with one sentence per line.")
    
    parser.add_argument("--model", default='bert-large-cased', type=str, help="Name of model.")
    parser.add_argument("--filter-layers",nargs='*', default=[], type=int, help="Filter out bias in layers.")
    parser.add_argument("--keep-information", action="store_true", help="Whether to keep the information dimensions.")
    parser.add_argument("--filter-threshold", default=1e-4, type=float, help="Threshold for bias filter.")

    args = parser.parse_args()
    
    all_correct = 0
    all_predicted = 0
    mlm, tokenizer = get_model_tokenizer(args.model, args.filter_layers,
                                         keep_information=args.keep_information, filter_threshold=args.filter_threshold)

    in_fn = os.path.join(args.data_dir, args.evaluation_file + ".sents")
    with open(in_fn, 'r') as in_file:
        for sent in tqdm(in_file):
            sent_correct, sent_predicted = prediction_test(mlm, tokenizer, sent.strip())
            all_correct += sent_correct
            all_predicted += sent_predicted
            
    out_fn = os.path.join(args.data_dir, get_name(args) + ".txt")
    with open(out_fn, 'w') as out_file:
        # out_file.write(f"Model: {args.model}\nFiltered layers: {args.filter_layers}\nKeep information: {args.keep_information}\nFilter threshold: {args.filter_threshold}"))
        out_file.write(f"MLM Accuracy: {all_correct/all_predicted}\nN predictions: {all_predicted}")

