import tensorflow as tf
from tqdm import tqdm
import os
import json
import argparse
from collections import defaultdict, Mapping
import sys


from transformers import BertTokenizer, BertConfig, TFBertModel
from transformers import RobertaTokenizer, TFRobertaModel
from transformers import XLMRobertaTokenizer, TFXLMRobertaModel

central_storage_strategy = tf.distribute.experimental.CentralStorageStrategy()

sys.path.append(r"/Users/tomasz/PycharmProjects/bias_vs_information/src")
from data_wrappers.wino_wrapper import WinoWrapper
import constants

def merge_dict(d1, d2):
    """
    Modifies d1 in-place to contain values from d2.  If any value
    in d1 is a dictionary (or dict-like), *and* the corresponding
    value in d2 is also a dictionary, then merge them in-place.
    """
    for k,v2 in d2.items():
        v1 = d1.get(k) # returns None if v1 has no value for this key
        if ( isinstance(v1, Mapping) and
                isinstance(v2, Mapping) ):
            merge_dict(v1, v2)
        else:
            d1[k] = v2


class TFRecordWrapper:
    
    modes = ['train', 'dev', 'test']
    data_map_fn = "data_map.json"

    def __init__(self, models):
        self.models = models
    
        self.map_tfrecord = dict()
        for mode in self.modes:
            self.map_tfrecord[mode] = dict()
            for model in models:
                self.map_tfrecord[mode][model] = None

    def _from_json(self, data_dir):
        with open(os.path.join(data_dir,self.data_map_fn),'r') as in_json:
            in_dict = json.load(in_json)
        for attribute, value in in_dict.items():
            if attribute == "models":
                present_value = self.__getattribute__(attribute)
                self.__setattr__(attribute, list(set(present_value + value)))
            elif attribute == "map_tfrecord":
                merge_dict(self.map_tfrecord, value)
    
    def _to_json(self, data_dir):
        out_dict = {"models": self.models,
                    "map_tfrecord": self.map_tfrecord}
        
        with open(os.path.join(data_dir,self.data_map_fn), 'w') as out_json:
            json.dump(out_dict, out_json, indent=2, sort_keys=True)


class TFRecordWriter(TFRecordWrapper):
    
    def __init__(self, models, data_dir, split_by_profession=False):

        super().__init__(models)
        if os.path.isfile(os.path.join(data_dir, self.data_map_fn)):
            self._from_json(data_dir)

        self.split_by_profession =split_by_profession
        self.model2tfrs = defaultdict(set)
        self.tfr2mode = dict()

        for mode in self.modes:
            for model in models:

                tfr_fn = self.struct_tfrecord_fn(model, mode, self.split_by_profession)
                self.map_tfrecord[mode][model] = tfr_fn

                self.model2tfrs[model].add(tfr_fn)
                self.tfr2mode[tfr_fn] = mode

    def compute_and_save(self, data_dir):
        
        for model_path in self.models:
            # This is crude, but should work
            do_lower_case = "uncased" in model_path
            model, tokenizer = self.get_model_tokenizer(model_path, do_lower_case=do_lower_case)
            for tfrecord_file in self.model2tfrs[model_path]:
                if os.path.isfile(os.path.join(data_dir, tfrecord_file)):
                    print(f"File {os.path.join(data_dir, tfrecord_file)} already exists, skipping!")
                    continue
                
                mode = self.tfr2mode[tfrecord_file]
                in_datasets = WinoWrapper(f"{data_dir}/en.txt", tokenizer, split_by_profession=self.split_by_profession)
                indices, all_wordpieces, all_segments, all_token_len, prof_positions, pronoun_positions,\
                m_biases, f_biases, m_informations, f_informations, objects, _ = in_datasets.training_examples(mode)
                
                options = tf.io.TFRecordOptions()#compression_type='GZIP')
                with tf.io.TFRecordWriter(os.path.join(data_dir, tfrecord_file), options=options) as tf_writer:
                    for (idx, wordpieces, segments, token_len, prof_pos, pron_pos, m_bias, f_bias, m_info, f_info, obj) in \
                            tqdm(zip(indices, tf.unstack(all_wordpieces), tf.unstack(all_segments), tf.unstack(all_token_len),
                                tf.unstack(prof_positions), tf.unstack(pronoun_positions), tf.unstack(m_biases), tf.unstack(f_biases),
                                tf.unstack(m_informations), tf.unstack(f_informations), tf.unstack(objects)),
                                desc="Embedding computation"):
                        # Computing embedding (difference between original and featured embedding)
                        emb_prof_base = self.calc_embeddings(model, wordpieces[1], segments[1], token_len[1], prof_pos)
                        emb_pron_base = self.calc_embeddings(model, wordpieces[1], segments[1], token_len[1], pron_pos)
                        emb_prof_feat = self.calc_embeddings(model, wordpieces[2], segments[2], token_len[2], prof_pos)
                        emb_pron_feat = self.calc_embeddings(model, wordpieces[3], segments[3], token_len[3], pron_pos)
                        
                        emb_prof_diff = [emb_f - emb_b for emb_f, emb_b in zip(emb_prof_feat, emb_prof_base)]
                        emb_pron_diff = [emb_f - emb_b for emb_f, emb_b in zip(emb_pron_feat, emb_pron_base)]
                        
                        train_example = self.serialize_example(idx, emb_prof_diff, emb_pron_diff, m_bias, f_bias, m_info, f_info, obj)
                        tf_writer.write(train_example.SerializeToString())
        self._to_json(data_dir)
    
    @staticmethod
    def struct_tfrecord_fn(model,mode, split_by_profession=False):
        if split_by_profession:
            fn = f"{model}_gendervec_{mode}_spb.tfrecord"
        else:
            fn = f"{model}_gendervec_{mode}.tfrecord"
        return fn
    
    @staticmethod
    def get_model_tokenizer(model_path, do_lower_case, seed=42):
        if model_path.startswith('bert'):
            tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=do_lower_case)
            model = TFBertModel.from_pretrained(model_path, output_hidden_states=True, output_attentions=False)
        elif model_path.startswith('roberta'):
            tokenizer = RobertaTokenizer.from_pretrained(model_path, do_lower_case=do_lower_case, add_prefix_space=True)
            model = TFRobertaModel.from_pretrained(model_path, output_hidden_states=True, output_attentions=False)
        elif model_path.startswith('jplu/tf-xlm-roberta'):
            tokenizer = XLMRobertaTokenizer.from_pretrained(model_path, do_lower_case=do_lower_case)
            model = TFXLMRobertaModel.from_pretrained(model_path, output_hidden_states=True, output_attentions=False)
        elif model_path.startswith('random-bert'):
            tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=True)
            config = BertConfig(seed=seed, output_hidden_states=True, output_attentions=False)
            model = TFBertModel(config)
        else:
            raise ValueError(f"Unknown Transformer name: {model_path}. "
                             f"Please select one of the supported models: {constants.SUPPORTED_MODELS}")
        
        return model, tokenizer
    
    @staticmethod
    def calc_embeddings(model, wordpieces, segments, max_token_len, position):
        wordpieces = tf.expand_dims(wordpieces, 0)
        segments = tf.expand_dims(segments, 0)
        max_token_len = tf.constant(max_token_len, shape=(1,), dtype=tf.int64)

        model_output = model(wordpieces, attention_mask=tf.sign(wordpieces), training=False)
        embeddings = model_output.hidden_states[1:]
        
        # average wordpieces to obtain word representation
        # cut to max nummber of words in batch, note that batch.max_token_len is a tensor, bu all the values are the same

        embeddings = [tf.map_fn(lambda x: tf.math.unsorted_segment_mean(x[0], x[1], x[2]),
                                (emb, segments, max_token_len), dtype=tf.float32) for emb in embeddings]

        # picking the embedding of the profession name
        embeddings_prof_mask = [emb[0,position,:] for emb in embeddings]
        return embeddings
    
    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    @staticmethod
    def serialize_example(idx, embeddings_prof, embeddings_pron, m_bias, f_bias, m_info, f_info, obj):
        feature = {'index': TFRecordWriter._int64_feature(idx),
                   'm_bias': TFRecordWriter._int64_feature(m_bias),
                   'f_bias': TFRecordWriter._int64_feature(f_bias),
                   'm_information': TFRecordWriter._int64_feature(m_info),
                   'f_information': TFRecordWriter._int64_feature(f_info),
                   'is_object': TFRecordWriter._int64_feature(obj)}
        feature.update({f'layer_{idx}_profession': TFRecordWriter._bytes_feature(tf.io.serialize_tensor(layer_embedding))
                        for idx, layer_embedding in enumerate(embeddings_prof)})
        feature.update({f'layer_{idx}_pronoun': TFRecordWriter._bytes_feature(tf.io.serialize_tensor(layer_embedding))
                        for idx, layer_embedding in enumerate(embeddings_pron)})

        return tf.train.Example(features=tf.train.Features(feature=feature))


class TFRecordReader(TFRecordWrapper):
    
    def __init__(self, data_dir, model_name):
        super().__init__([])
        self.data_dir = data_dir
        self.model_name = model_name
        self._from_json(data_dir)
        TFRecordReader.parse_factory(self.model_name)
    
    def read(self, read_tasks, read_languages):
        if self.model_name not in self.models:
            raise ValueError(f"Data for this model are not available in the directory: {self.model_name}\n"
                             f" supported models: {self.models}")
        
        for mode in self.modes:
            tfr_fn = os.path.join(self.data_dir, self.map_tfrecord[mode][self.model_name])
            data_set = tf.data.TFRecordDataset(tfr_fn,
                                               #compression_type='GZIP',
                                               buffer_size=constants.BUFFER_SIZE)

            self.__setattr__(mode, data_set)
    
    @staticmethod
    def parse(example):
        pass
    
    @classmethod
    def parse_factory(cls, model_name):
        
        def parse(example):
            features_dict = {"index": tf.io.FixedLenFeature([], tf.int64),
                             'm_bias': tf.io.FixedLenFeature([], tf.bool),
                             'f_bias': tf.io.FixedLenFeature([], tf.bool),
                             'm_information': tf.io.FixedLenFeature([], tf.bool),
                             'f_information': tf.io.FixedLenFeature([], tf.bool),
                             'is_object': tf.io.FixedLenFeature([], tf.bool)
                             }
            features_dict.update({f"layer_{idx}_profession": tf.io.FixedLenFeature([], tf.string)
                                  for idx in range(constants.MODEL_LAYERS[model_name])})

            features_dict.update({f"layer_{idx}_pronoun": tf.io.FixedLenFeature([], tf.string)
                                  for idx in range(constants.MODEL_LAYERS[model_name])})

            example = tf.io.parse_single_example(example, features_dict)
            
            return example
        
        cls.parse = parse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("data_dir", help="Directory with data")
    parser.add_argument("--models", nargs='*', default=['bert-base-cased'], type=str, help="List of models.")
    parser.add_argument("--sbp", action='store_true', help="Split by profession.")

    args = parser.parse_args()

    TFRW = TFRecordWriter(args.models, args.data_dir, split_by_profession=args.sbp)
    TFRW.compute_and_save(args.data_dir)