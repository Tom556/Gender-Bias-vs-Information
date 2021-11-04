import numpy as np
import tensorflow as tf
import json
from nltk.tokenize import word_tokenize
from collections import defaultdict

import constants

RANDOM_SEED = 2021


class JsonWrapper():

	def __init__(self, json_file, tokenizer, split_by_profession=False):

		self.json_name = json_file
		self.tokenizer = tokenizer

		self.tokens = []
		self.wordpieces = []

		self.ortho_forms = []

		self.positions = []

		self.biases = []
		self.informations = []
		self.if_objects = []

		self.splits = dict()

		self.read_json(json_file)

		if split_by_profession:
			proportion = (0.6, 0.2, 0.2)
			self.split_dataset(proportion=proportion, split_by_profession=split_by_profession)
		else:
			self.split_dataset(split_by_profession=split_by_profession)

	def read_json(self, json_file):

		with open(json_file, 'r') as in_json:
			data = json.load(in_json)

		for example in data:
			self.tokens.append(word_tokenize(example["sent"]))
			self.ortho_forms.append(example["orth"])

			self.positions.append(example["position"])

			self.biases.append(example["gender_bias"])
			self.informations.append(example["gender_information"])

			self.if_objects.append(example["object"])

	def get_bert_ids(self, wordpieces):
		"""Token ids from Tokenizer vocab"""
		token_ids = self.tokenizer.convert_tokens_to_ids(wordpieces)
		input_ids = token_ids + [0] * (constants.MAX_WORDPIECES - len(wordpieces))
		return input_ids

	def split_dataset(self, proportion=(0.8, 0.1, 0.1), modes=('train', 'dev', 'test'), split_by_profession=True):

		all_indices = []
		removed_indices = []

		profession2indices = defaultdict(list)

		for idx, sent_tokens in enumerate(self.tokens[:]):
			sent_wordpieces = [self.tokenizer.cls_token] + self.tokenizer.tokenize((' '.join(sent_tokens))) + \
			                  [self.tokenizer.sep_token]
			self.wordpieces.append(sent_wordpieces)

			if len(sent_tokens) >= constants.MAX_TOKENS:
				print(f"Sentence {idx} too many tokens, in file {self.json_name}, skipping.")
				removed_indices.append(idx)

			elif len(sent_wordpieces) >= constants.MAX_WORDPIECES:
				print(f"Sentence {idx} too many wordpieces, in file {self.json_name}, skipping.")
				removed_indices.append(idx)
			else:
				wordpiece_pointer = 1

				for token in sent_tokens:
					worpieces_per_token = len(self.tokenizer.tokenize(token))
					wordpiece_pointer += worpieces_per_token

				if wordpiece_pointer + 1 != len(sent_wordpieces):
					print(f'Sentence {idx} mismatch in number of tokens, in file {self.json_name}, skipping.')
					removed_indices.append(idx)
				else:
					all_indices.append(idx)
					profession2indices[self.ortho_forms[idx]].append(idx)



		if split_by_profession:
			self.splits = defaultdict(list)
			unique_professions = list(set(self.ortho_forms))
			split_modes = np.split(np.array(unique_professions),
			                       [int(prop * len(unique_professions)) for prop in np.cumsum(proportion)])

			for split, mode in zip(split_modes, modes):
				for profession in split:
					self.splits[mode].extend(profession2indices[profession])
		else:
			split_modes = np.split(np.array(all_indices),
			                       [int(prop * len(all_indices)) for prop in np.cumsum(proportion)])
			for split, mode in zip(split_modes, modes):
				self.splits[mode] = list(split)

		self.splits["removed"] = removed_indices

	def training_examples(self, mode):
		'''
		Joins wordpices of tokens, so that they correspond to the tokens in conllu file.
		:param shuffle: whether to shuffle tokens in each sentence

		:return:
			2-D tensor  [num valid sentences, max num wordpieces] bert wordpiece ids,
			2-D tensor [num valid sentences, max num wordpieces] wordpiece to word segment mappings
			1-D tensor [num valid sentences] number of words in each sentence
			1-D tensor [num valid sentences] positions of the profession word
			1-D tensor [num valid sentences] biases boolean if female biased
			1-D tensor [num valid sentences] informations boolean if factual female gender
			1-D tensor [num valid sentences] objects boolean if profession is in object
		'''

		if mode not in self.splits:
			raise ValueError(f"Unkown split of dataset: {mode}, possible modes: {self.splits.keys()}")
		indices = self.splits[mode]

		if not self.wordpieces:
			self.split_dataset()

		segments = []
		max_segment = []
		bert_ids = []
		for idx in indices:
			sent_tokens = self.tokens[idx]
			sent_wordpieces = self.wordpieces[idx]

			sent_segments = np.zeros((constants.MAX_WORDPIECES,), dtype=np.int64) - 1
			segment_id = 0
			wordpiece_pointer = 1

			for token in sent_tokens:
				worpieces_per_token = len(self.tokenizer.tokenize(token))
				sent_segments[wordpiece_pointer:wordpiece_pointer + worpieces_per_token] = segment_id
				wordpiece_pointer += worpieces_per_token
				segment_id += 1

			segments.append(tf.constant(sent_segments, dtype=tf.int64))
			bert_ids.append(tf.constant(self.get_bert_ids(sent_wordpieces), dtype=tf.int64))
			max_segment.append(segment_id)

		return tf.stack(bert_ids), tf.stack(segments), tf.constant(max_segment, dtype=tf.int64), \
		       tf.constant(np.array(self.positions)[indices], dtype=tf.int64), \
		       tf.constant(np.array(self.biases)[indices],dtype=tf.float32), \
		       tf.constant(np.array(self.informations)[indices], dtype=tf.float32), \
		       tf.constant(np.array(self.if_objects)[indices], dtype=tf.float32)
