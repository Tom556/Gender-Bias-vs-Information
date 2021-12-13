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

		self.prof_positions = []
		self.pronoun_positions = []
		self.pronoun_counters = []

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
				prof_position = int(split_line[1])
				object = (prof_position > 1)
				tokens = word_tokenize(split_line[2])
				pronoun_position = []
				pronoun_counter = 0
				
				ortho = split_line[3]
				
				for tidx, tok in enumerate(tokens):
					if tok in constants.male_pronouns:
						if gender == 'male':
							pronoun_position.append(tidx)
					elif tok in constants.female_pronouns:
						if gender == 'female':
							pronoun_position.append(tidx)
					elif tok in constants.neutral_pronouns:
						if gender == 'neutral':
							pronoun_position.append(tidx)
					else:
						continue
					pronoun_counter += 1
				
				self.tokens.append(tokens)
				self.ortho_forms.append(ortho)
				self.prof_positions.append(prof_position)
				self.pronoun_positions.append(pronoun_position)
				self.pronoun_counters.append(pronoun_counter)
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

	def get_model_ids(self, wordpieces):
		"""Token ids from Tokenizer vocab"""
		token_ids = self.tokenizer.convert_tokens_to_ids(wordpieces)
		input_ids = token_ids + [0] * (constants.MAX_WORDPIECES - len(wordpieces))
		return np.array(input_ids)

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
				
			elif self.pronoun_counters[idx] != 1:
				print(f"Sentence {idx} : {' '.join(self.tokens[idx])} number of pronouns does not equal to 1.")
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
			random.shuffle(all_indices)					
			split_modes = np.split(np.array(all_indices),
			                       [int(prop * len(all_indices)) for prop in np.cumsum(proportion)])
			for split, mode in zip(split_modes, modes):
				self.splits[mode] = list(split)
				random.shuffle(self.splits[mode])

		self.splits["removed"] = removed_indices


	#TODO: change return to structured output
	def training_examples(self, mode):
		if mode not in self.splits:
			raise ValueError(f"Unkown split of dataset: {mode}, possible modes: {self.splits.keys()}")
		indices = self.splits[mode]

		if not self.wordpieces:
			self.split_dataset()

		segments = []
		max_segment = []
		bert_ids = []
		
		masked_wordpieces = []
		for idx in indices:
			sent_tokens = self.tokens[idx]
			sent_wordpieces = self.wordpieces[idx]
			
			prof_pos = self.prof_positions[idx]
			pronoun_pos = self.pronoun_positions[idx]

			sent_segments = np.zeros((4, constants.MAX_WORDPIECES,), dtype=np.int64) - 1
			sent_bert_ids = np.zeros((4, constants.MAX_WORDPIECES,), dtype=np.int64)
			sent_max_segment = np.zeros((4,), dtype=np.int64)
			sent_masked_wordpieces = [[], [], [], []]
			
			# sent variants:
			# 0 : original sentence
			# 1 : professions and pronouns masked
			# 2 : only professions masked
			# 3 : only pronouns masked
			for sent_variant in range(4):
				segment_id = 0
				wordpiece_pointer = 1
				sent_wordpieces_modified = [self.tokenizer.cls_token]

				for tidx, token in enumerate(sent_tokens):
					if tidx == prof_pos and sent_variant in (1, 2):
						token = self.tokenizer.mask_token
						sent_masked_wordpieces[sent_variant].append(wordpiece_pointer)
					elif tidx in pronoun_pos and sent_variant in (1,3):
						token = self.tokenizer.mask_token
						sent_masked_wordpieces[sent_variant].append(wordpiece_pointer)
					wordpieces = self.tokenizer.tokenize(token)
					worpieces_per_token = len(wordpieces)
					
					sent_wordpieces_modified.extend(wordpieces)
					sent_segments[sent_variant, wordpiece_pointer:wordpiece_pointer + worpieces_per_token] = segment_id
					
					wordpiece_pointer += worpieces_per_token
					
					segment_id += 1
				
				sent_wordpieces_modified.append(self.tokenizer.sep_token)
				if sent_variant == 0:
					assert sent_wordpieces_modified == sent_wordpieces, f"expected: {sent_wordpieces}, obtained: {sent_wordpieces_modified}"
				
				sent_bert_ids[sent_variant,:] = self.get_model_ids(sent_wordpieces_modified)
				sent_max_segment[sent_variant] = segment_id

			segments.append(tf.constant(sent_segments, dtype=tf.int64))
			bert_ids.append(tf.constant(sent_bert_ids, dtype=tf.int64))
			max_segment.append(tf.constant(sent_max_segment, dtype=tf.int64))
			
			masked_wordpieces.append(sent_masked_wordpieces)
		
		first_pronoun_positions = [pronoun_position[0] for pronoun_position in self.pronoun_positions]

		return tf.stack(indices), tf.stack(bert_ids), tf.stack(segments), tf.stack(max_segment), \
		       tf.constant(np.array(self.prof_positions)[indices], dtype=tf.int64), \
			   tf.constant(np.array(first_pronoun_positions)[indices], dtype=tf.int64), \
		       tf.constant(np.array(self.m_biases)[indices],dtype=tf.bool), \
			   tf.constant(np.array(self.f_biases)[indices], dtype=tf.bool), \
		       tf.constant(np.array(self.m_informations)[indices], dtype=tf.bool), \
			   tf.constant(np.array(self.f_informations)[indices], dtype=tf.bool), \
		       tf.constant(np.array(self.if_objects)[indices], dtype=tf.bool), \
			   masked_wordpieces
