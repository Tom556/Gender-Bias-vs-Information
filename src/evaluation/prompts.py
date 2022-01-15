import tensorflow as tf
import numpy as np
import csv
from tqdm import tqdm
import json
from copy import copy
import os
import argparse

from utils import get_mlm_tokenizer
import constants
from data_wrappers import wino_wrapper


prompts = {"IS": (3, "male\t{position}\tHe is {det}{profession}.\t{l_profession}\n"),
           "WAS": (3, "male\t{position}\tHe was {det}{profession}.\t{l_profession}\n"),
           "JOB": (4, "male\t{position}\tHis job is {det}{profession}.\t{l_profession}\n"),
           "WORKS": (4, "male\t{position}\tHe works as {det}{profession}.\t{l_profession}\n"),
           "LIKES": (1, "male\t{position}\t{det}{profession} said that he likes his job.\t{l_profession}\n"),
           "HATES": (1, "male\t{position}\t{det}{profession} said that he hates his job.\t{l_profession}\n")}

non_professional = {'child', 'teenager', 'onlooker', 'victim', 'protester', 'taxpayer', 'homeowner', 'owner',
                    'employee', 'visitor', 'guest', 'bystander', 'client', 'witness', 'buyer',
                    'pedestrian', 'someone', 'resident', 'customer', 'passenger', 'patient' }


def calculate_probability(probs, pronoun, tokenizer, model_path):
    do_lower_case = ('uncased' in model_path or 'albert' in model_path or 'electra' in model_path)
    if pronoun.lower() == "he":
        if do_lower_case:
            m_tok_ids = tokenizer.convert_tokens_to_ids(["he"])
            f_tok_ids = tokenizer.convert_tokens_to_ids(["she"])
        else:
            m_tok_ids = tokenizer.convert_tokens_to_ids(["He", "he"])
            f_tok_ids = tokenizer.convert_tokens_to_ids(["She","she"])
        if "albert" in model_path:
            m_tok_ids += tokenizer.convert_tokens_to_ids([u"▁he"])
            f_tok_ids += tokenizer.convert_tokens_to_ids([u"▁she"])

    elif pronoun.lower() == "his":
        if do_lower_case:
            m_tok_ids = tokenizer.convert_tokens_to_ids(["his"])
            f_tok_ids = tokenizer.convert_tokens_to_ids(["her"])
        else:
            m_tok_ids = tokenizer.convert_tokens_to_ids(["His", "his"])
            f_tok_ids = tokenizer.convert_tokens_to_ids(["Her", "her"])

        if "albert" in model_path:
            m_tok_ids += tokenizer.convert_tokens_to_ids([u"▁his"])
            f_tok_ids += tokenizer.convert_tokens_to_ids([u"▁her"])

    assert tokenizer.unk_token_id not in m_tok_ids
    assert tokenizer.unk_token_id not in f_tok_ids

    m_prob = tf.reduce_sum(tf.gather(probs, m_tok_ids)).numpy()
    f_prob = tf.reduce_sum(tf.gather(probs, f_tok_ids)).numpy()
    
    return np.log(m_prob) - np.log(f_prob), m_prob + f_prob

def get_name(args):
    name = f"empirical_bias_{args.model}"
    
    if args.filter_layers:
        fl_str = '_'.join(list(map(str,args.filter_layers)))
        name += f"_f{fl_str}"
        
        if args.filter_threshold != 1e-4:
            name += f"_thr_{str(args.filter_threshold)}"
        
        if args.keep_information:
            name += "_keep-information"
        if args.no_intercept:
            name += "_no-intercept"
    return name


def write_to_csv(data, args): 
    fields = ['profession', 'bias', 'gender', 'professional']
    fields+= ["IS", "WAS", "JOB", "WORKS", "LIKES1", "LIKES2", "HATES1", "HATES2"]
   
    file_name = get_name(args) + ".csv"
    output_file = os.path.join(args.data_dir, file_name)
    with open(output_file, "w") as csv_file:
        writer = csv.DictWriter(csv_file, fields)
        writer.writeheader()
        for k, v in data.items():
            row = {'profession': k}
            row.update(**v)
            writer.writerow(row)


def write_to_json_txt(data, args):
    
    gendered_MSE = 0.
    ungendered_MSE = 0.
    gendered_ABS = 0.
    ungendered_ABS = 0.
    gendered_MEAN = 0.
    ungendered_MEAN = 0.
    
    n_gendered = 0
    n_ungendered = 0
    out_dict = {}
    for k, v in data.items():
        if k == 'TOTAL_PROB':
            continue
        TEs = [v[prompt_type] for prompt_type in ("IS", "WAS", "JOB", "WORKS", "LIKES1", "LIKES2", "HATES1", "HATES2")]
        mean_TE = np.array(TEs).mean()
        out_dict[k] = str(mean_TE)
        
        if k in constants.male_gendered or k in constants.female_gendered:
            n_gendered += 1
            
            gendered_MSE += mean_TE ** 2.0
            gendered_ABS += np.abs(mean_TE)
            gendered_MEAN += mean_TE
            
        else:
            n_ungendered += 1

            ungendered_MSE += mean_TE ** 2.0
            ungendered_ABS += np.abs(mean_TE)
            ungendered_MEAN += mean_TE

    gendered_STD = np.sqrt((gendered_MSE / n_gendered) - (gendered_MEAN / n_gendered) ** 2.0)
    ungendered_STD = np.sqrt((ungendered_MSE / n_ungendered) - (ungendered_MEAN / n_ungendered)** 2.0)
    

    file_name = get_name(args) + ".json"
    output_file = os.path.join(args.data_dir, file_name)
    with open(output_file, 'w') as out_json:
        json.dump(out_dict, out_json, indent=2, sort_keys=True)

    file_name = get_name(args) + ".txt"
    output_file = os.path.join(args.data_dir, file_name)
    with open(output_file, 'w') as out_txt:
        out_txt.write(f"MSE: gendered {gendered_MSE/n_gendered} ungendered {ungendered_MSE/n_ungendered}\n"
                      f"ABS: gendered {gendered_ABS/n_gendered} ungendered {ungendered_ABS/n_ungendered}\n"
                      f"MEAN: gendered {gendered_MEAN/n_gendered} ungendered {ungendered_MEAN/n_ungendered}\n"
                      f"STD: gendered {gendered_STD} ungendered {ungendered_STD}\n"
                      f"COUNT: gendered {n_gendered} ungendered {n_ungendered}\n")
    
    
def generate_simple_prompts(args):
    for prompt_type, (position, prompt_line) in prompts.items():
        if os.path.isfile(os.path.join(args.data_dir, prompt_type + "_prompts.txt")):
            continue
        output_file = os.path.join(args.data_dir, prompt_type + "_prompts.txt")

        with open(output_file, "w") as of:
            for profession in constants.male_biased | constants.female_biased | constants.male_gendered | constants.female_gendered | constants.non_biased:
                if profession == "someone":
                    det = ""
                elif prompt_type in ("IS", "WAS"):
                    det = "the "
                elif prompt_type in ("JOB", "WORKS") and profession[0] in 'aeiou':
                    det = "an "
                elif prompt_type in ("JOB", "WORKS"):
                    det = "a "
                else:
                    det = "The "

                if profession == "someone":
                    if position == 1:
                        of.write(prompt_line.format(position=position-1, det=det, profession="Someone", l_profession=profession))
                    else:
                        of.write(prompt_line.format(position=position-1, det=det, profession=profession, l_profession=profession))
                else:
                    of.write(prompt_line.format(position=position, det=det, profession=profession, l_profession=profession))
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("data_dir", help="Directory with data")
    parser.add_argument("--model", default='roberta-large', type=str, help="Name of model.")
    parser.add_argument("--filter-layers",nargs='*', default=[], type=int, help="Filter out bias in layers.")
    parser.add_argument("--keep-information", action="store_true", help="Whether to keep the information dimensions.")
    parser.add_argument("--remove-information", action="store_true", help="Whether to remove the information dimensions.")
    parser.add_argument("--filter-threshold", default=1e-12, type=float, help="Threshold for bias filter.")
    parser.add_argument("--no-intercept", action="store_true", help="Whether to don't use intercept in filtering")
    args = parser.parse_args()
    
    generate_simple_prompts(args)

    mlm, tokenizer = get_mlm_tokenizer(args.model, filter_layers=args.filter_layers,
                                       keep_information=args.keep_information, filter_threshold=args.filter_threshold)


    data = {}
    for prof in constants.male_biased:
        assert prof not in data
        data[prof] = {'bias': 1, 'gender': 0, 'professional': 1}
    for prof in constants.female_biased:
        assert prof not in data
        data[prof] = {'bias': -1, 'gender': 0, 'professional': 1}
    for prof in constants.non_biased:
        assert prof not in data
        if prof in non_professional:
            data[prof] = {'bias': 0, 'gender': 0, 'professional': 0}
        else:
            data[prof] = {'bias': 0, 'gender': 0, 'professional': 1}

    for prof in constants.male_gendered:
        assert prof not in data
        data[prof] = {'bias': 0, 'gender': 1, 'professional': 1}

    for prof in constants.female_gendered:
        assert prof not in data
        data[prof] = {'bias': 0, 'gender': -1, 'professional': 1}
        
    data["TOTAL_PROB"] = {'bias': 0, 'gender': 0, 'professional': 0}


    for prompt_type in prompts.keys():
       
        input_file = output_file = os.path.join(args.data_dir, prompt_type + "_prompts.txt")
        wrapper = wino_wrapper.WinoWrapper(input_file, tokenizer)

        for prof in wrapper.ortho_forms:
            assert prof in data, f"{prof} not in data!"
    
        for mode in ('train', 'dev', 'test'):
            indices, all_wordpieces, all_segments, all_token_len, all_prof_positions, all_pronoun_positions, \
            m_biases, f_biases, m_informations, f_informations, objects, all_masked_positions = wrapper.training_examples(mode)
            
            for index, sent_wordpieces, sent_masked_pos in tqdm(zip(indices, tf.unstack(all_wordpieces), all_masked_positions)):
                
                prof = wrapper.ortho_forms[index]
                pron = wrapper.pronoun_ortho_forms[index][0]
                if prompt_type in ("HATES", "LIKES"):
                    pron_2 = wrapper.pronoun_ortho_forms[index][1]
                
                
                base_wordpieces = tf.expand_dims(sent_wordpieces[1,:], 0)
                biased_wordpieces = tf.expand_dims(sent_wordpieces[3,:], 0)

                if prompt_type in ("HATES", "LIKES"):
                    base_masked_pron = sent_masked_pos[1][1]
                    biased_masked_pron = sent_masked_pos[3][0]

                    base_masked_pron_2 = sent_masked_pos[1][2]
                    biased_masked_pron_2 = sent_masked_pos[3][1]
                else:
                    base_masked_pron = sent_masked_pos[1][0]
                    biased_masked_pron = sent_masked_pos[3][0]
                
                if args.filter_layers:
                    # dirty hack
                    mlm.layers[0].encoder.with_intercept = False
                    output_base = mlm(base_wordpieces)

                    mlm.layers[0].encoder.with_intercept = (True and not args.no_intercept)
                    output_biased = mlm(biased_wordpieces)
                else:
                    output_base = mlm(base_wordpieces)
                    output_biased = mlm(biased_wordpieces)
                
                prob_base = tf.nn.softmax(output_base.logits[0,base_masked_pron,:])
                prob_biased = tf.nn.softmax(output_biased.logits[0,biased_masked_pron,:])

                rel_base, sum_base = calculate_probability(prob_base, pron, tokenizer, args.model)
                rel_biased, _ = calculate_probability(prob_biased, pron, tokenizer, args.model)

                TE = rel_biased - rel_base

                if prompt_type in ("HATES", "LIKES"):
                    prob_base_2 = tf.nn.softmax(output_base.logits[0,base_masked_pron_2,:])
                    prob_biased_2 = tf.nn.softmax(output_biased.logits[0,biased_masked_pron_2,:])

                    rel_base_2, sum_base_2 = calculate_probability(prob_base_2, pron_2, tokenizer, args.model)
                    rel_biased_2, _ = calculate_probability(prob_biased_2, pron_2, tokenizer, args.model)

                    TE_2 = rel_biased_2 - rel_base_2

                    data[prof][prompt_type + "1"] = TE
                    data[prof][prompt_type + "2"] = TE_2
                else:
                    data[prof][prompt_type] = TE
    
            if prompt_type in ("HATES", "LIKES"):
                data["TOTAL_PROB"][prompt_type + "1"] = sum_base
                data["TOTAL_PROB"][prompt_type + "2"] = sum_base_2
            else:
                data["TOTAL_PROB"][prompt_type] = sum_base
                

            
    write_to_csv(data, args)
    write_to_json_txt(data, args)
            
