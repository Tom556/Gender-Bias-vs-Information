import argparse
from collections import defaultdict
import csv
import os
from nltk.tokenize import word_tokenize
import tensorflow as tf
from tqdm import tqdm

from utils import get_mlm_tokenizer, get_outfile_name

pronoun2gender = {'he': 'male', 'him': 'male', 'his': 'male', 'she': 'female', 'her': 'female'}

# Following list copied from: https://github.com/google-research-datasets/gap-coreference
# Fieldnames used in the gold dataset .tsv file.
GOLD_FIELDNAMES = [
    'ID', 'Text', 'Pronoun', 'Pronoun-offset', 'A', 'A-offset', 'A-coref', 'B',
    'B-offset', 'B-coref', 'URL'
]


# Following function highly inspired on: https://github.com/google-research-datasets/gap-coreference
def read_annotations(filename):
    """Reads coreference annotations for the examples in the given file.
    """
    
    fieldnames = GOLD_FIELDNAMES

    with open(filename, 'rU') as f:
        reader = csv.DictReader(f, fieldnames=fieldnames, delimiter='\t')
        
        # Skip the header line in the gold data
        next(reader, None)
        
        for r_idx, row in enumerate(reader):
            sent = row['Text']
            pronoun_offset = int(row['Pronoun-offset'])
            pronoun = row['Pronoun']
            tokens = sent.strip().split(" ")
            
            char_index = 0
            skip = False
            for token_index, token in enumerate(tokens):
                if char_index == pronoun_offset:
                    token_parts = word_tokenize(token)
                    token_clean = token_parts[0]
                    if len(token_parts) > 1:
                        token_ending = "".join(token_parts[1:])
                    else:
                        token_ending = ""
                    if token_clean != pronoun:
                        print(f"Skipping {r_idx} expected: {pronoun}, got: {token}")
                        skip = True
                    break
                char_index += len(token) + 1
            if skip:
                continue
        
            yield tokens, pronoun, token_index, token_ending


def calculate_probability(probs, pronoun, tokenizer):
    do_lower_case = tokenizer.do_lower_case
    if pronoun.lower() == "he" or pronoun.lower() == "she":
        if do_lower_case:
            m_tok_ids = tokenizer.convert_tokens_to_ids(["he"])
            f_tok_ids = tokenizer.convert_tokens_to_ids(["she"])
        else:
            m_tok_ids = tokenizer.convert_tokens_to_ids(["He", "he"])
            f_tok_ids = tokenizer.convert_tokens_to_ids(["She","she"])
    
    elif pronoun.lower() == "his":
        if do_lower_case:
            m_tok_ids = tokenizer.convert_tokens_to_ids(["his"])
            f_tok_ids = tokenizer.convert_tokens_to_ids(["her"])
        else:
            m_tok_ids = tokenizer.convert_tokens_to_ids(["His", "his"])
            f_tok_ids = tokenizer.convert_tokens_to_ids(["Her", "her"])

    elif pronoun.lower() == "him":
        if do_lower_case:
            m_tok_ids = tokenizer.convert_tokens_to_ids(["him"])
            f_tok_ids = tokenizer.convert_tokens_to_ids(["her"])
        else:
            m_tok_ids = tokenizer.convert_tokens_to_ids(["Him", "him"])
            f_tok_ids = tokenizer.convert_tokens_to_ids(["Her", "her"])

    elif pronoun.lower() == "her":
        if do_lower_case:
            m_tok_ids = tokenizer.convert_tokens_to_ids(["him", "his"])
            f_tok_ids = tokenizer.convert_tokens_to_ids(["her"])
        else:
            m_tok_ids = tokenizer.convert_tokens_to_ids(["Him", "him", "His", "his"])
            f_tok_ids = tokenizer.convert_tokens_to_ids(["Her", "her"])
    else:
        raise ValueError(f"Unrecognized pronoun: {pronoun}")
    
    assert tokenizer.unk_token_id not in m_tok_ids
    assert tokenizer.unk_token_id not in f_tok_ids
    
    m_prob = tf.reduce_sum(tf.gather(probs, m_tok_ids)).numpy()
    f_prob = tf.reduce_sum(tf.gather(probs, f_tok_ids)).numpy()
    
    return m_prob, f_prob

    
def gap_test(model, tokenizer, sent):
    
    tokens = [tokenizer.cls_token] + tokenizer.tokenize(sent) + [tokenizer.sep_token]
    if len(tokens) > 256:
        print(f"Skipping sent: {sent}\nToo many tokens")
        return 0, 0
    
    mask_index = tokens.index(tokenizer.mask_token)
    
    token_ids = tf.expand_dims(tf.constant(tokenizer.convert_tokens_to_ids(tokens), dtype=tf.int64), 0)
    outputs = model(token_ids)
    probabilities = tf.nn.softmax(outputs.logits[0,mask_index,:])
    
    return calculate_probability(probabilities, pronoun, tokenizer)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("data_dir", help="Directory with data")
    parser.add_argument("--evaluation-file", default='gap-validation', type=str, help="File with one sentence per line.")
    parser.add_argument("--model", default='bert-large-cased', type=str, help="Name of model.")
    parser.add_argument("--filter-layers",nargs='*', default=[], type=int, help="Filter out bias in layers.")
    parser.add_argument("--keep-information", action="store_true", help="Whether to keep the information dimensions.")
    parser.add_argument("--filter-threshold", default=1e-8, type=float, help="Threshold for bias filter.")
    
    args = parser.parse_args()

    mlm, tokenizer = get_mlm_tokenizer(args.model, args.filter_layers,
                                       keep_information=args.keep_information, filter_threshold=args.filter_threshold)
    
    in_fn = os.path.join(args.data_dir, args.evaluation_file + ".tsv")

    correct_male = 0
    correct_female = 0
    all_male = 0
    all_female = 0
    for sent_idx, (tokens, pronoun, token_index, token_ending) in tqdm(enumerate(read_annotations(in_fn))):
        gender = pronoun2gender.get(pronoun.lower(), None)
        if not gender:
            print(f"Unsuported pronoun: {pronoun.lower()}")
            continue

        tokens[token_index] = tokenizer.mask_token + token_ending
        m_prob, f_prob = gap_test(mlm, tokenizer, " ".join(tokens))
        if not m_prob or not f_prob:
            continue
        
        if gender == 'male':
            all_male += 1
            if m_prob > f_prob:
                correct_male += 1
        elif gender == 'female':
            all_female += 1
            if m_prob < f_prob:
                correct_female += 1
                
    out_fn = os.path.join(args.data_dir, "gap_accuracy_" + get_outfile_name(args) + ".txt")
    with open(out_fn, 'w') as out_file:
        out_file.write(f"Accuracy: {(correct_male+correct_female)/(all_male+all_female)}\n"
                       f"Male: {correct_male/all_male}\nFemale: {correct_female/all_female}\n"
                       f"Number of examples male: {all_male} female: {all_female}\n")