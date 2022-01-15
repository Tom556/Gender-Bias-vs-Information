import os
import argparse
import json

from data_wrappers.tfrecord_wrapper import TFRecordReader
from network import Network
from reporting.reporter import AccuracyReporter, CorrelationReporter

import constants

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #
    parser.add_argument("parent_dir", type=str, default="../experiments", help="Parent experiment directory")
    parser.add_argument("data_dir", type=str, default='../data',
                        help="Directory where tfrecord files are stored")
    parser.add_argument("--json-data", type=str, default=None, help="JSON with conllu and languages for training")
    
    parser.add_argument("--languages", nargs='*', default=['en'], type=str,
                        help="Languages to probe.")
    parser.add_argument("--tasks", nargs='*', type=str,
                        help="Probing tasks (gender information or bias)")
    parser.add_argument("--objects-only", action="store_true",
                        help="Whether to only consider the embeddings of professions that are objects of a sentence"
                             "(hence, they are closer to preposition with gender information)")
    parser.add_argument("--repeat", default=1, type=int,
                        help="How many times iterate trough trianing examples in one epoch")

# Probe arguments
    parser.add_argument("--probe-rank", default=None, type=int, help="Rank of the probe")
    parser.add_argument("--layer-index", default=6, type=int, help="Index of BERT's layer to probe."
                                                                   "If -1 all layers embeddings are averaged")
    # Train arguments
    parser.add_argument("--seed", default=42, type=int, help="Seed for variable initialisation")
    parser.add_argument("--batch-size", default=10, type=int, help="Batch size")
    parser.add_argument("--epochs", default=40, type=int, help="Maximal number of training epochs")
    parser.add_argument("--learning-rate", default=0.02, type=float, help="Initial learning rate")
    parser.add_argument("--ortho", default=0.2, type=float,
                        help="Orthogonality reguralization (SRIP) for language map matrices.")
    parser.add_argument("--l1", default=None, type=float, help="L1 reguralization of the weights.")
    parser.add_argument("--clip-norm", default=1.5, type=float, help="Clip gradient norm to this value")
    # parser.add_argument("--gate-threshold", default=None, type=float, help="Gate masking large values in the probe")

    parser.add_argument("--subsample-train", default=None, type=int,
                        help="Size of subsample taken from a training set.")

    # Specify Bert Model
    parser.add_argument("--model",
                        default=f"bert-{constants.SIZE_BASE}-{constants.LANGUAGE_MULTILINGUAL}-{constants.CASING_CASED}",
                        help="Transformer model name (see: https://huggingface.co/transformers/pretrained_models.html)")

    args = parser.parse_args()

    if not args.probe_rank:
        args.probe_rank = constants.MODEL_DIMS[args.model]

    do_lower_case = (constants.CASING_UNCASED in args.model)

    experiment_name = f"task_{'_'.join(args.tasks)}-layer_{args.layer_index}-trainl_{'_'.join(args.languages)}"

    if args.learning_rate != 0.02:
        experiment_name += f"-lr_{args.learning_rate}"
    if args.batch_size != 10:
        experiment_name += f"-bs_{args.batch_size}"
    if args.clip_norm != 1.5:
        experiment_name += f"-cn_{args.clip_norm}"
    if args.seed != 42:
        experiment_name += f"-seed_{args.seed}"
    if args.repeat != 1:
        experiment_name += f"-rep_{args.repeat}"
    if args.ortho != 0.05:
        experiment_name += f"-or_{args.ortho}"
    if args.l1 is not None and args.l1 != 0:
        experiment_name += f"-l1_{args.l1}"
    # experiment_name += '-abs_loss'
    # experiment_name += '-no-square-in-loss'
    # if args.gate_threshold:
    #     experiment_name += f"-gt_{args.gate_threshold}"
    if args.objects_only:
        experiment_name += '-objects_only'
        
    args.out_dir = os.path.join(args.parent_dir, experiment_name)
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    tf_reader = TFRecordReader(args.data_dir, args.model)
    tf_reader.read(args.tasks, args.languages)

    network = Network(args)
    network.load(args)
    
    for mode in ("test", "dev", "train"):
        reporter = AccuracyReporter(args, network, args.tasks, getattr(tf_reader, mode), mode)
        reporter.compute(args)
        reporter.write(args)

        reporter = CorrelationReporter(args, network, args.tasks, getattr(tf_reader, mode), mode)
        reporter.compute(args)
        reporter.write(args)
