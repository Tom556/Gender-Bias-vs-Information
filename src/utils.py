import numpy as np

from transformers import BertTokenizer, TFBertForMaskedLM, TFBertModel
from transformers import RobertaTokenizer, TFRobertaForMaskedLM, TFRobertaModel
from transformers import XLNetTokenizer, TFXLNetLMHeadModel
from transformers import AlbertTokenizer, TFAlbertForMaskedLM
from transformers import ElectraTokenizer, TFElectraForMaskedLM

from tf_modified_bert_encoder import TFModifiedBertEncoder
import constants


def get_mlm_tokenizer(model_path, filter_layers=None, keep_information=False, filter_threshold=1e-4):
    do_lower_case = ('uncased' in model_path)
    if model_path.startswith('bert'):
        tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=do_lower_case)
        model = TFBertForMaskedLM.from_pretrained(model_path, output_hidden_states=False, output_attentions=False)
        if filter_layers:
            OSPs_out = np.load(f'../experiments/{model_path}-intercept/osp_bias_{model_path}.npz', allow_pickle=True)
            if keep_information:
                OSPs_in = np.load(f'../experiments/{model_path}-intercept/osp_information_{model_path}.npz', allow_pickle=True)
            else:
                OSPs_in = None
            encoder = model.layers[0].encoder
            modified_encoder = TFModifiedBertEncoder(OSPs_out, filter_layers,
                                                     projection_matrices_in=OSPs_in, filter_threshold=filter_threshold, source=encoder)
            model.layers[0].encoder = modified_encoder
    elif model_path.startswith('roberta'):
        tokenizer = RobertaTokenizer.from_pretrained(model_path, do_lower_case=do_lower_case, add_prefix_space=True)
        model = TFRobertaForMaskedLM.from_pretrained(model_path, output_hidden_states=False, output_attentions=False)
    elif model_path.startswith('xlnet'):
        tokenizer = XLNetTokenizer.from_pretrained(model_path, do_lower_case=do_lower_case)
        model = TFXLNetLMHeadModel.from_pretrained(model_path, output_hidden_states=False, output_attentions=False)
    elif model_path.startswith('albert'):
        tokenizer = AlbertTokenizer.from_pretrained(model_path, do_lower_case=True, remove_spaces=True)
        model = TFAlbertForMaskedLM.from_pretrained(model_path, output_hidden_states=False, output_attentions=False)
    elif model_path.startswith('electra'):
        tokenizer = ElectraTokenizer.from_pretrained("google/" + model_path, do_lower_case=True, remove_spaces=True)
        model = TFElectraForMaskedLM.from_pretrained("google/" + model_path, output_hidden_states=False, output_attentions=False)
    else:
        raise ValueError(f"Unknown Transformer name: {model_path}. "
                         f"Please select one of the supported models: {constants.SUPPORTED_MODELS}")
    
    return model, tokenizer


def get_outfile_name(args):
    name = f"{args.model}_{args.evaluation_file}"
    if args.filter_layers:
        fl_str = '_'.join(list(map(str,args.filter_layers)))
        name += f"_f{fl_str}"
        
        if args.filter_threshold != 1e-4:
            name += f"_thr_{str(args.filter_threshold)}"
        
        if args.keep_information:
            name += "_keep-information"
    
    return name