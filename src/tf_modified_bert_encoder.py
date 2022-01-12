import tensorflow as tf
import numpy as np
from transformers.models.bert.modeling_tf_bert import TFBertEncoder, TFBaseModelOutput


class TFModifiedBertEncoder(TFBertEncoder):
    def __init__(self, projection_matrices_out, layers_to_modify,
                 projection_matrices_in=None, source=None, filter_threshold=1e-4, **kwargs):
        # copying all fields from source
        if source is not None:
            self.__dict__.update(source.__dict__)
        else:
            super().__init__(**kwargs)

        self.filter_vectors = []
        self.intercepts = []
        self.orthogonal_transformations = []
        self.layers_to_modify = layers_to_modify
        self.with_intercept = False
        
        self.precompute_filters(projection_matrices_out, projection_matrices_in, filter_threshold)
        
    def precompute_filters(self, projection_matrices_out, projection_matrices_in, filter_threshold):
        self.filter_vectors = []
        self.intercepts = []
        self.orthogonal_transformations = []
        for layer_idx, (scaling_vector, orthogonal_transformation, intercept) in enumerate(projection_matrices_out):
    
            scaling_vector_out = tf.squeeze(tf.constant(scaling_vector, dtype=tf.float32))
            orthogonal_transformation = tf.constant(orthogonal_transformation, dtype=tf.float32)
            intercept = tf.squeeze(tf.constant(intercept, dtype=tf.float32))

            filter_vector = tf.cast((tf.abs(scaling_vector_out) <= filter_threshold), dtype=tf.float32)
            intercept = (intercept * tf.abs(scaling_vector_out) * (1.0 - filter_vector))
            if projection_matrices_in is not None:
                scaling_vector_in, _, _ = projection_matrices_in[layer_idx]
                scaling_vector_in = tf.squeeze(tf.constant(scaling_vector_in, dtype=tf.float32))
                filter_vector = 1.0 - ((1.0 - filter_vector) * tf.cast(tf.abs(scaling_vector_out) > tf.abs(scaling_vector_in), dtype=tf.float32))
            
            self.filter_vectors.append(filter_vector)
            self.intercepts.append(intercept)
            self.orthogonal_transformations.append(orthogonal_transformation)

    def transform_hidden_states(self, hidden_states, layer_number):
        filter_vector = self.filter_vectors[layer_number]
        orthogonal_transformation = self.orthogonal_transformations[layer_number]
        intercept = self.intercepts[layer_number]
        
        if self.with_intercept:
            projected_states = tf.matmul((tf.matmul(hidden_states, orthogonal_transformation) * filter_vector) + intercept, tf.transpose(orthogonal_transformation))
        else:
            projected_states = tf.matmul(tf.matmul(hidden_states, orthogonal_transformation) * filter_vector, tf.transpose(orthogonal_transformation))
        return projected_states

    def call(
            self,
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
            output_hidden_states,
            return_dict,
            training=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], output_attentions, training=training
            )
            hidden_states = layer_outputs[0]
            if i in self.layers_to_modify:
                hidden_states = self.transform_hidden_states(hidden_states, i)
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
