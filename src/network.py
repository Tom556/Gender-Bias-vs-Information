"""Classes for training and running inference on probes."""
import os

import tensorflow as tf
from abc import abstractmethod
from functools import partial

from tqdm import tqdm
import numpy as np
import resource

import constants

class Network():
    
    ONPLATEU_DECAY = 0.1
    ES_PATIENCE = 4
    ES_DELTA = 1e-35
    
    PENALTY_THRESHOLD = 1.5
    INITIAL_EPOCHS = 3
    
    class Probe():
        def __init__(self, args):
            print('Constructing Probe')
            self.probe_rank = args.probe_rank
            self.model_dim = constants.MODEL_DIMS[args.model]
            self.languages = args.languages
            self.tasks = args.tasks

            self._orthogonal_reg = args.ortho
            
            self.average_layers = (args.layer_index == -1)
            
            self.OrthogonalTransformations = {lang: tf.Variable(tf.initializers.Identity(gain=1.0)((self.model_dim, self.probe_rank)),
                                                   trainable=True, name='{}_map'.format(lang))
                                 for lang in self.languages}

            if self._orthogonal_reg:
                # if orthogonalization is used multilingual probe is only diagonal scaling
                self.BinaryProbe = {task: tf.Variable(tf.random_uniform_initializer(minval=-0.5, maxval=0.5, seed=args.seed)
                                                        ((1, self.probe_rank,)),
                                                        trainable=True, name=f'{task}_probe', dtype=tf.float32)
                                      for task in self.tasks}
            else:
                raise NotImplementedError


            self.LayerWeights = {f'lw_{task}':
                                     tf.Variable(tf.initializers.Ones()((1,constants.MODEL_LAYERS[args.model],1,1)),
                                                 trainable=self.average_layers, name='{}_layer_weights'.format(task))
                                 for task in args.tasks}

            self._train_fns = {lang: {task: self.train_factory(lang, task)
                                      for task in self.tasks}
                               for lang in self.languages}
            
            self._layer_idx = args.layer_index
            self._lr = args.learning_rate
            self._clip_norm = args.clip_norm
            
            self._l1_reg = args.l1
            self._optimizer = tf.optimizers.Adam(lr=self._lr)
            
            self._writer = tf.summary.create_file_writer(args.out_dir, flush_millis=10 * 1000, name='summary_writer')
        
        
        @staticmethod
        @tf.function
        def ortho_reguralization(w):
            """(D)SO implementation according to:
            https://papers.nips.cc/paper/7680-can-we-gain-more-from-orthogonality-regularizations-in-training-deep-networks.pdf"""
            #
            w_rows = w.shape[0]
            w_cols = w.shape[1]
            
            reg = tf.norm(tf.transpose(w) @ w - tf.eye(w_cols)) + tf.norm(w @ tf.transpose(w) - tf.eye(w_rows))
            # to avoid NaN in gradient update
            if reg == 0:
                reg = 1e-6
            return reg
        
        def decrease_lr(self, decay_factor):
            self._lr *= decay_factor
            self._optimizer.learning_rate.assign(self._lr)

        @tf.function
        def get_projections(self, embeddings, language, task):
            """ Computes projections after Orthogonal Transformation, and after Dimension Scaling"""

            orthogonal_projections = embeddings @ self.OrthogonalTransformations[language]
            if self._orthogonal_reg:
                projections = orthogonal_projections * self.BinaryProbe[task]
            else:
                projections = embeddings @ self.BinaryProbe[task]
    
            return orthogonal_projections, projections

        @tf.function
        def _forward(self, embeddings, language, task, embeddings_gate=None):
            """ Computes all n^2 pairs of distances after projection
            for each sentence in a batch.

            Note that due to padding, some distances will be non-zero for pads.
            Computes (B(h_i-h_j))^T(B(h_i-h_j)) for all i,j
            """
            if self.average_layers:
                embeddings = tf.reduce_mean(embeddings * self.LayerWeights[f'lw_{task}'], axis=1, keepdims=False)
            _, projections = self.get_projections(embeddings,  language, task)
            if embeddings_gate is not None:
                projections = projections * embeddings_gate

            projections = tf.expand_dims(projections, 1)  # shape [batch, 1, emb_dim]
            transposed_projections = tf.transpose(projections, perm=(1, 0, 2))  # shape [1, batch, emb_dim]
            diffs = projections - transposed_projections  # shape [batch, batch, emb_dim]
            squared_diffs = tf.reduce_sum(tf.math.square(diffs), axis=-1) # shape [batch, batch]
            return squared_diffs
        
        @tf.function
        def _compute_target_mask(self, feature_vector):
            batch_size = tf.shape(feature_vector)[0]
            
            feature_vector = tf.expand_dims(feature_vector, 1) # shape [batch, 1]
            transposed_feature_vector = tf.transpose(feature_vector) # shape [1, batch]
    
            target = tf.math.logical_xor(feature_vector, transposed_feature_vector)
            target = tf.cast(target, dtype=tf.float32)
            # mask is upper triangular matrix
            mask = tf.ones_like(target, dtype=tf.float32) - tf.eye(batch_size, dtype=tf.float32)
            mask = tf.linalg.band_part(mask, 0, -1)
            
            return target, mask

        @tf.function
        def _loss(self, predicted_distances, gold_distances, mask):
            loss = tf.reduce_sum(tf.abs(predicted_distances - gold_distances) * mask) / \
                            tf.clip_by_value(tf.reduce_sum(mask), 1., constants.MAX_TOKENS ** 2.)
            return loss

        def train_factory(self, language, task):
            # separate train function is needed to avoid variable creation on non-first call
            # see: https://github.com/tensorflow/tensorflow/issues/27120
            @tf.function(experimental_relax_shapes=True)
            def train_on_batch(bias, information, is_object, embeddings, l1_lambda=0.0):
        
                with tf.GradientTape() as tape:
                    if 'bias' in task:
                        target, mask = self._compute_target_mask(bias)
                    else:
                        target, mask = self._compute_target_mask(information)

                    predicted_distances = self._forward(embeddings, language, task)

                    loss = self._loss(predicted_distances, target, mask)
                    if self._orthogonal_reg:
                        ortho_penalty = self.ortho_reguralization(self.OrthogonalTransformations[language])
                        loss += self._orthogonal_reg * ortho_penalty
                    if self._l1_reg:
                        probe_l1_penalty = tf.norm(self.BinaryProbe[task], ord=1)
                        loss += l1_lambda * probe_l1_penalty
        
                variables = [self.BinaryProbe[task], self.OrthogonalTransformations[language]]
        
                if self.average_layers:
                    variables.append(self.LayerWeights[f'lw_{task}'])
        
                gradients = tape.gradient(loss, variables)
                gradient_norms = [tf.norm(grad) for grad in gradients]
                if self._clip_norm:
                    gradients = [tf.clip_by_norm(grad, self._clip_norm) for grad in gradients]
                self._optimizer.apply_gradients(zip(gradients, variables))
                tf.summary.experimental.set_step(self._optimizer.iterations)
        
                with self._writer.as_default(), tf.summary.record_if(self._optimizer.iterations % 20 == 0
                                                                           or self._optimizer.iterations == 1):
                    tf.summary.scalar("train/batch_loss_{}".format(language), loss)
                    tf.summary.scalar("train/probe_gradient_norm", gradient_norms[0])
                    tf.summary.scalar("train/{}_map_gradient_norm".format(language), gradient_norms[1])

                    if self._orthogonal_reg:
                        tf.summary.scalar("train/{}_nonorthogonality_penalty".format(language), ortho_penalty)
                    if self._l1_reg:
                        tf.summary.scalar("train/probe_l1_penalty", probe_l1_penalty)
                        tf.summary.scalar("train/l1_lambda", l1_lambda)
                    tf.summary.scalar("train/{}_map_gradient_norm".format(language), gradient_norms[1])
        
                return loss
            return train_on_batch

        @tf.function(experimental_relax_shapes=True)
        def evaluate_on_batch(self, bias, information, is_object, embeddings, language, task):
            if 'bias' in task:
                target, mask = self._compute_target_mask(bias)
            else:
                target, mask = self._compute_target_mask(information)
            
            predicted_distances = self._forward(embeddings, language, task)
            loss = self._loss(predicted_distances, target, mask)
            return loss

        @tf.function(experimental_relax_shapes=True)
        def predict_on_batch(self, embeddings, language, task):
            predicted_distances = self._forward(embeddings, language, task)
            return predicted_distances

    def __init__(self, args):

        self.languages = args.languages
        self.tasks = args.tasks

        self.probe = self.Probe(args)

        self.optimal_loss = np.inf

        # Checkpoint managment:
        self.ckpt = tf.train.Checkpoint(optimizer=self.probe._optimizer,
                                        **self.probe.OrthogonalTransformations,
                                        **self.probe.LayerWeights,
                                        **self.probe.BinaryProbe)
        self.checkpoint_manager = tf.train.CheckpointManager(self.ckpt, os.path.join(args.out_dir, 'params'),
                                                             max_to_keep=1)

    @staticmethod
    def decode(serialized_example, layer_idx, model):
        features_to_decode = {"index": tf.io.FixedLenFeature([], tf.int64),
                              'bias': tf.io.FixedLenFeature([], tf.int64),
                              'information': tf.io.FixedLenFeature([], tf.int64),
                              'is_object': tf.io.FixedLenFeature([], tf.int64)
                              }
        if layer_idx == -1:
            features_to_decode.update({f'layer_{idx}': tf.io.FixedLenFeature([], tf.string)
                                       for idx in range(constants.MODEL_LAYERS[model])})
        else:
            features_to_decode[f'layer_{layer_idx}'] = tf.io.FixedLenFeature([], tf.string)
    
        x = tf.io.parse_example(
            serialized_example,
            features= features_to_decode)
    
        index = tf.cast(x["index"], dtype=tf.int64)
        bias = tf.cast(x["bias"], dtype=tf.bool)
        information = tf.cast(x["information"], dtype=tf.bool)
        is_object = tf.cast(x["is_object"], dtype=tf.bool)
        if layer_idx == -1:
            embeddings = [tf.io.parse_tensor(x[f"layer_{idx}"], out_type=tf.float32)
                          for idx in range(constants.MODEL_LAYERS[model])]
            embeddings = tf.stack(embeddings, axis=0)
    
        else:
            embeddings = tf.io.parse_tensor(x[f"layer_{layer_idx}"], out_type=tf.float32)
    
        return index, bias, information, is_object, embeddings

    @staticmethod
    def data_pipeline(tf_data, languages, tasks, args, mode='train'):
        # TODO: Read correctly gender data
    
        datasets_to_interleve = []
        for langs in languages:
            for task in tasks:
                data = tf_data
                data = data.map(partial(Network.decode, layer_idx=args.layer_index, model=args.model),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
                if args.layer_index >= 0:
                    data = data.cache()
                if mode == 'train':
                    data = data.shuffle(constants.SHUFFLE_SIZE, args.seed)
                data = data.batch(args.batch_size)
                data = data.map(lambda *x: (langs, task, x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
                data = data.prefetch(tf.data.experimental.AUTOTUNE)
                datasets_to_interleve.append(data)

        if len(datasets_to_interleve) == 0:
            return []
        if len(datasets_to_interleve) == 1:
            return datasets_to_interleve[0]
    
        return tf.data.experimental.sample_from_datasets(datasets_to_interleve)

    def train(self, tf_reader, args):
        curr_patience = 0
        train = self.data_pipeline(tf_reader.train, self.languages, self.tasks, args, mode='train')
        dev = {lang: {task: self.data_pipeline(tf_reader.dev, [lang], [task], args, mode='dev')
                      for task in self.tasks}
               for lang in self.languages}
    
        l1_lambda = 0.0
    
        for epoch_idx in range(args.epochs):
        
            progressbar = tqdm(enumerate(train))
        
            for batch_idx, (lang, task, batch) in progressbar:
                lang = lang.numpy().decode()
                task = task.numpy().decode()
            
                _, batch_bias, batch_information, batch_is_object, batch_embeddings = batch

                batch_loss = self.probe._train_fns[lang][task](batch_bias, batch_information, batch_is_object, batch_embeddings, l1_lambda)
            
                progressbar.set_description(f"Training, batch loss: {batch_loss:.4f}, memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /1024 / 1024:.2f}")
        
            eval_loss = self.evaluate(dev, 'validation', args)
            if eval_loss < self.optimal_loss - self.ES_DELTA:
                self.optimal_loss = eval_loss
                self.checkpoint_manager.save()
                curr_patience = 0
            else:
                curr_patience += 1
        
            if curr_patience > 0:
                self.probe.decrease_lr(self.ONPLATEU_DECAY)
            if curr_patience > self.ES_PATIENCE:
                self.load(args)
                break
        
            if self.probe._l1_reg and not l1_lambda:
                sum_ortho_penalty = 0.
                for orthogonal_matrix in self.probe.OrthogonalTransformations.values():
                    sum_ortho_penalty += self.probe.ortho_reguralization(orthogonal_matrix)
            
                if sum_ortho_penalty <= self.PENALTY_THRESHOLD:
                    # Turn on l1 regularization when orthogonality penalty is already small
                    l1_lambda = self.probe._l1_reg
                    # Loss equation changes, so we need to reset optimal loss
                    self.optimal_loss = np.inf
        
            with self.probe._writer.as_default():
                tf.summary.scalar("train/learning_rate", self.probe._optimizer.learning_rate)

    def evaluate(self, data, data_name, args):
        all_losses = np.zeros((len(self.languages), len(self.tasks)))
        for lang_idx, language in enumerate(self.languages):
            for task_idx, task in enumerate(self.tasks):
            
                progressbar = tqdm(enumerate(data[language][task]))
                for batch_idx, (_, _, batch) in progressbar:
                    _, batch_bias, batch_information, batch_is_object, batch_embeddings = batch

                    batch_loss = self.probe.evaluate_on_batch(batch_bias, batch_information, batch_is_object, batch_embeddings,language, task)

                    progressbar.set_description(f"Evaluating on {language} {task} loss: {batch_loss:.4f}")
                    all_losses[lang_idx][task_idx] += batch_loss
            
                all_losses[lang_idx][task_idx] = all_losses[lang_idx][task_idx] / (batch_idx + 1.)
                with self.probe._writer.as_default():
                    tf.summary.scalar("{}/loss_{}_{}".format(data_name, language, task), all_losses[lang_idx][task_idx])
            
                print(f'{data_name} loss on {language} {task} : {all_losses[lang_idx][task_idx]:.4f}')
    
        with self.probe._writer.as_default():
            tf.summary.scalar("{}/loss".format(data_name), all_losses.mean())
        return all_losses.mean()

    def load(self, args):
        self.checkpoint_manager.restore_or_initialize()

    

