import os
import tensorflow as tf
import argparse

import constants
import numpy as np
from itertools import combinations


def load_ckpt(layer_idx, dim=1024, model_path='bert-large-cased'):
    languages = ["en"]
    out_dir = f'../experiments/{model_path}-intercept/task_bias_information-layer_{layer_idx}-trainl_en-cn_1.0-or_0.1'

    model_dim = dim
    probe_rank = dim
    
    BinaryProbe = {task: tf.Variable(tf.random_uniform_initializer(minval=-0.5, maxval=0.5)
                                     ((1,probe_rank)),
                                     trainable=False, name=f'{task}_probe', dtype=tf.float32)
                   for task in ['bias', 'information']}
    Intercept = {f'intercept_{task}': tf.Variable(tf.initializers.Zeros()((1, probe_rank)),
                                                  trainable=True, name=f'{task}_intercept', dtype=tf.float32) for task in ['bias', 'information']}
    
    optimizer=tf.optimizers.Adam()
    
    OrthogonalTransformations = {lang: tf.Variable(tf.initializers.Identity(gain=1.0)((model_dim, model_dim)),
                                                   trainable=False, name='{}_map'.format(lang))
                                 for lang in languages}
    
    
    ckpt = tf.train.Checkpoint(optimizer=optimizer,**OrthogonalTransformations, **BinaryProbe, **Intercept)
    
    checkpoint_manger = tf.train.CheckpointManager(ckpt, os.path.join(out_dir, 'params'), max_to_keep=1)
    
    checkpoint_manger.restore_or_initialize()
    
    return ckpt

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("model_path", type=str, help="Model for which to save the probe parameters.")
    
    args = parser.parse_args()
    
    for task in ('information', 'bias'):
        OSPs = []
        for layer_idx in range(constants.MODEL_LAYERS[args.model_path]):
            ckpt = load_ckpt(layer_idx, dim=constants.MODEL_DIMS[args.model_path], model_path=args.model_path)
            OSPs.append((getattr(ckpt, task).numpy(),ckpt.en.numpy(), getattr(ckpt, f"intercept_{task}").numpy()))
            print((np.abs(getattr(ckpt, task).numpy()) > 1e-12).sum())
        OSPs = np.array(OSPs)

        out_fn = f'../experiments/{args.model_path}-intercept/osp_{task}_{args.model_path}.npz'
        
        
        with open(out_fn, 'wb') as out_file:
            np.save(out_file, OSPs)
            print(f"{task} probe parameters saved to {out_fn}")
