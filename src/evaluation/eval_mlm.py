import tensorflow as tf
import numpy as np
import csv
from tqdm import tqdm
import json
from copy import copy
import os
import argparse

from utils import get_mlm_tokenizer, get_outfile_name


def prediction_test(model, tokenizer, sent):
    sent_correct = 0
    sent_predicted = 0
    
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
    mlm, tokenizer = get_mlm_tokenizer(args.model, args.filter_layers,
                                       keep_information=args.keep_information, filter_threshold=args.filter_threshold)

    in_fn = os.path.join(args.data_dir, args.evaluation_file + ".sents")
    with open(in_fn, 'r') as in_file:
        for sent in tqdm(in_file):
            sent_correct, sent_predicted = prediction_test(mlm, tokenizer, sent.strip())
            all_correct += sent_correct
            all_predicted += sent_predicted
            
    out_fn = os.path.join(args.data_dir, "mlm_accuracy_" + get_outfile_name(args) + ".txt")
    with open(out_fn, 'w') as out_file:
        # out_file.write(f"Model: {args.model}\nFiltered layers: {args.filter_layers}\nKeep information: {args.keep_information}\nFilter threshold: {args.filter_threshold}"))
        out_file.write(f"MLM Accuracy: {all_correct/all_predicted}\nN predictions: {all_predicted}\n")

