import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
from abc import abstractmethod
from collections import defaultdict

from network import Network


class Metric:
    
    def __init__(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def reset_state(self):
        pass
    
    @abstractmethod
    def update_state(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def result(self):
        pass
    
    
class Accuracy(Metric):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        
        self.all_correct = 0
        self.prediction_count = 0
    
        self.tp = 0
        self.all_pos = 0
        self.all_pred = 0
        
        super().__init__()

    def __call__(self,gold, predicted, mask):
        self.update_state(gold.numpy(), predicted.numpy(), mask.numpy())
    
    def reset_state(self):
        self.all_correct = 0
        self.prediction_count = 0
        
        self.tp = 0
        
        self.all_pos = 0
        self.all_pred = 0
    
    def update_state(self, predicted, gold, mask):
        self.prediction_count += np.sum(mask)
        correct = (gold.astype(bool) == (predicted >= self.threshold))
        self.all_correct += np.sum(correct & mask.astype(bool))
        
        self.tp += np.sum(correct & gold.astype(bool) & mask.astype(bool))
        
        self.all_pos += np.sum(mask.astype(bool) & gold.astype(bool))
        self.all_pred += np.sum((predicted >= self.threshold) & mask.astype(bool))

    def result(self):
        
        recall = self.tp / self.all_pos
        precision = self.tp / self.all_pred
        f1 = 2. * (recall * precision) / (recall + precision)
        if not self.prediction_count:
            return 0., 0., 0., 0.
        return self.all_correct / self.prediction_count, recall, precision, f1
    

class Reporter():
    
    def __init__(self, args, network, dataset, dataset_name):
        self.network = network
        self.dataset = dataset
        self.dataset_name = dataset_name

    def predict(self, args, lang, task):
        data_pipe = Network.data_pipeline(self.dataset, [lang], [task], args, mode=self.dataset_name)
        progressbar = tqdm(enumerate(data_pipe), desc="Predicting, {}, {}".format(lang, task))
        for batch_idx, (_, _, batch) in progressbar:
            _, batch_m_bias, batch_f_bias, batch_m_information, batch_f_information, batch_is_object, batch_embeddings = batch
            predicted = self.network.probe.predict_on_batch(batch_embeddings, lang, task)

            if task == 'm_bias':
                feature_vector = batch_m_bias
            elif task == 'f_bias':
                feature_vector = batch_f_bias
            elif task == 'm_information':
                feature_vector = batch_m_information
            elif task == 'f_information':
                feature_vector = batch_f_information
            else:
                raise ValueError(f"Unrecognized task: {task}")
            
            gold, mask = self.network.probe._compute_target_mask(feature_vector)

            yield predicted, gold, mask
            
            
class AccuracyReporter(Reporter):
    def __init__(self, args, network, tasks, dataset, dataset_name, acc_threshold=0.5):
        super().__init__(args, network, dataset, dataset_name)
        
        self._languages = args.languages
        self._tasks = tasks
        self.accuracy_d = defaultdict(dict)
        
        self.threshold = acc_threshold

    def write(self, args):
        for lang in self._languages:
            for task in self._tasks:
                prefix = '{}.{}.{}.'.format(self.dataset_name, lang, task)
                if args.objects_only:
                    prefix += "objects_only."

                with open(os.path.join(args.out_dir, prefix + 'results'), 'w') as accuracy_f:
                    acc, m_acc, f_acc, f1 = self.accuracy_d[lang][task].result()
                    accuracy_f.write(f'acc: {acc}\n'
                                     f'recall: {m_acc}\n'
                                     f'precision: {f_acc}\n'
                                     f'f1: {f1}\n')

    def compute(self, args):
        for lang in self._languages:
            for task in self._tasks:
                self.accuracy_d[lang][task] = Accuracy(self.threshold)
                for pred_values, gold_values, mask in self.predict(args, lang, task):
                    self.accuracy_d[lang][task](pred_values, gold_values, mask)
