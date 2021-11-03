import os
import argparse
import json
import random


def read_data(file_name: str,pro_stereotypical: bool):
    examples = []
    with open(file_name, 'r') as infile:
        for line in infile:
            split_line = line.strip().split('\t')
            
            # 0 means male, 1 means famele
            gender = (split_line[0] == "female")
            bias = gender if pro_stereotypical else not gender
            object = int(split_line[1]) > 1

            
            example = {"sent": split_line[2],
                       "position": int(split_line[1]),
                       "object": object,
                       "orth": split_line[3],
                       "gender_information": gender,
                       "gender_bias": bias}
            examples.append(example)
            
    return examples
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("pro_data", type=str, help="File with pro-stereotypical data.")
    parser.add_argument("anti_data", type=str, help="file with anti-stereotypical data.")
    parser.add_argument("out_data", type=str, help="JSON where to save the data")
    
    args = parser.parse_args()
    examples = read_data(args.pro_data, pro_stereotypical=True)
    examples += read_data(args.anti_data, pro_stereotypical=False)
    
    random.shuffle(examples)
    with open(args.out_data, 'w') as outfile:
        json.dump(examples, outfile, indent=2)