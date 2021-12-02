import numpy as np
import tensorflow as tf
import random
import json
from nltk.tokenize import word_tokenize
from collections import defaultdict

import constants

RANDOM_SEED = 2021


class WinoWrapper():

	def __init__(self, wino_file, tokenizer, split_by_profession=False):

		#self.json_name = json_file
		self.wino_name = wino_file
		self.tokenizer = tokenizer

		self.tokens = []
		self.wordpieces = []

		self.ortho_forms = []

		self.positions = []

		self.m_biases = []
		self.f_biases = []
		
		self.m_informations = []
		self.f_informations = []
		
		self.if_objects = []

		self.splits = dict()

		self.read_wino(wino_file)

		if split_by_profession:
			proportion = (0.6, 0.2, 0.2)
			self.split_dataset(proportion=proportion, split_by_profession=split_by_profession)
		else:
			self.split_dataset(split_by_profession=split_by_profession)
			
	def read_wino(self, wino_file):
		with open(wino_file, 'r') as in_wino:
			for line in in_wino:
				split_line = line.strip().split('\t')
				
				# 0 means male, 1 means famele
				gender = split_line[0]
				position = int(split_line[1])
				object = (position > 1)
				tokens = word_tokenize(split_line[2])
				ortho = split_line[3]
				
				self.tokens.append(tokens)
				self.ortho_forms.append(ortho)
				self.positions.append(position)
				self.if_objects.append(object)
				
				if gender == 'male':
					self.m_informations.append(True)
					self.f_informations.append(False)
					
				elif gender == 'female':
					self.m_informations.append(False)
					self.f_informations.append(True)
				
				else:
					self.m_informations.append(False)
					self.f_informations.append(False)
					
				if ortho in constants.male_biased:
					self.m_biases.append(True)
					self.f_biases.append(False)
				
				elif ortho in constants.female_biased:
					self.m_biases.append(False)
					self.f_biases.append(True)
					
				else:
					self.m_biases.append(False)
					self.f_biases.append(False)

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
				print(f"Sentence {idx} too many tokens, in file {self.wino_name}, skipping.")
				removed_indices.append(idx)

			elif len(sent_wordpieces) >= constants.MAX_WORDPIECES:
				print(f"Sentence {idx} too many wordpieces, in file {self.wino_name}, skipping.")
				removed_indices.append(idx)
				
			elif self.ortho_forms[idx] in constants.problematic_list:
				print(f"Sentence {idx} problematic word: {self.ortho_forms[idx]}, in file {self.wino_name}, skipping.")
				removed_indices.append(idx)
			else:
				wordpiece_pointer = 1

				for token in sent_tokens:
					worpieces_per_token = len(self.tokenizer.tokenize(token))
					wordpiece_pointer += worpieces_per_token

				if wordpiece_pointer + 1 != len(sent_wordpieces):
					print(f'Sentence {idx} mismatch in number of tokens, in file {self.wino_name}, skipping.')
					removed_indices.append(idx)
				else:
					all_indices.append(idx)
					profession2indices[self.ortho_forms[idx]].append(idx)

		if split_by_profession:
			self.splits = defaultdict(list)
			
			# predefined splits sampled rendomly (to allow better comparability between models)
			if proportion == (0.6, 0.2, 0.2):
				split_modes = [constants.profession_splits[mode] for mode in modes]
			else:
				unique_professions = list(set(self.ortho_forms))
				split_modes = np.split(np.array(unique_professions), [int(prop * len(unique_professions)) for prop in np.cumsum(proportion)])
			
			for split, mode in zip(split_modes, modes):
				for profession in split:
					self.splits[mode].extend(profession2indices[profession])
				random.shuffle(self.splits[mode])
		else:
			split_modes = np.split(np.array(all_indices),
			                       [int(prop * len(all_indices)) for prop in np.cumsum(proportion)])
			for split, mode in zip(split_modes, modes):
				self.splits[mode] = list(split)
				random.shuffle(self.splits[mode])

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

		return tf.stack(indices), tf.stack(bert_ids), tf.stack(segments), tf.constant(max_segment, dtype=tf.int64), \
		       tf.constant(np.array(self.positions)[indices], dtype=tf.int64), \
		       tf.constant(np.array(self.m_biases)[indices],dtype=tf.bool), \
			   tf.constant(np.array(self.f_biases)[indices], dtype=tf.bool), \
		       tf.constant(np.array(self.m_informations)[indices], dtype=tf.bool), \
			   tf.constant(np.array(self.f_informations)[indices], dtype=tf.bool), \
		       tf.constant(np.array(self.if_objects)[indices], dtype=tf.bool)
