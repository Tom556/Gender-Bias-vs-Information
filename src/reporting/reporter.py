import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
from abc import abstractmethod
from collections import defaultdict
from scipy import stats

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
    def __init__(self):
        
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
        predictions = np.round(predicted) # np.round(np.clip(predicted, -1., 1.))
        gold = np.round(gold)
        correct = (gold == predictions)
        self.all_correct += np.sum(correct & mask.astype(bool))
        
        self.tp += np.sum(correct & gold.astype(bool) & mask.astype(bool))
        
        self.all_pos += np.sum(mask.astype(bool) & gold.astype(bool))
        self.all_pred += np.sum(predictions.astype(bool) & mask.astype(bool))

    def result(self):
        
        recall = self.tp / self.all_pos
        precision = self.tp / self.all_pred
        f1 = 2. * (recall * precision) / (recall + precision)
        if not self.prediction_count:
            return 0., 0., 0., 0.
        return self.all_correct / self.prediction_count, recall, precision, f1
    
    
class Correlation(Metric):
    def __init__(self, threshold=0.5):
        self.all_predicted = []
        self.all_gold = []
        
        super().__init__()

    def __call__(self,gold, predicted, mask):
        self.update_state(gold.numpy(), predicted.numpy(), mask.numpy())

    def reset_state(self):
        self.all_predicted = []
        self.all_gold = []

    def update_state(self, predicted, gold, mask):

        self.all_predicted.append(predicted[mask.astype(bool)])
        self.all_gold.append(gold[mask.astype(bool)])
    
    def result(self):
        
        predicted_array = np.concatenate(self.all_predicted)
        gold_array = np.concatenate(self.all_gold)
        
        scorr, _ = stats.spearmanr(predicted_array, gold_array)
        pcorr, _ = stats.pearsonr(predicted_array, gold_array)
        return scorr, pcorr
    
    
class Reporter():
    
    def __init__(self, args, network, dataset, dataset_name):
        self.network = network
        self.dataset = dataset
        self.dataset_name = dataset_name

    def predict(self, args, lang, task):
        data_pipe = Network.data_pipeline(self.dataset, [lang], [task], args, mode=self.dataset_name)
        progressbar = tqdm(enumerate(data_pipe), desc="Predicting, {}, {}".format(lang, task))
        for batch_idx, (_, _, batch) in progressbar:
            
            _, batch_m_bias, batch_f_bias, batch_m_information, batch_f_information, batch_emp_bias, batch_is_object, \
            batch_embeddings_prof, batch_embeddings_pron = batch

            if task == 'bias':
                feature_vector = batch_emp_bias # tf.cast(batch_m_bias, tf.int64) - tf.cast(batch_f_bias, tf.int64)
                batch_embeddings = batch_embeddings_pron
            elif task == 'information':
                feature_vector = tf.cast(batch_m_information, tf.int64) - tf.cast(batch_f_information, tf.int64)
                batch_embeddings = batch_embeddings_prof
            else:
                raise ValueError(f"Unrecognized task: {task}")

            predicted_pos, predicted_neg, predicted_abs = self.network.probe.predict_on_batch(batch_embeddings, lang, task)
            gold_pos, gold_neg, _, mask = self.network.probe._compute_target_mask(feature_vector)

            predictions = predicted_pos - predicted_neg
            yield predictions, gold_pos - gold_neg, mask
            
            
class AccuracyReporter(Reporter):
    def __init__(self, args, network, tasks, dataset, dataset_name):
        super().__init__(args, network, dataset, dataset_name)
        
        self._languages = args.languages
        self._tasks = tasks
        self.accuracy_d = defaultdict(dict)


    def write(self, args):
        for lang in self._languages:
            for task in self._tasks:
                prefix = '{}.{}.{}.acc.'.format(self.dataset_name, lang, task)
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
                self.accuracy_d[lang][task] = Accuracy()
                for pred_values, gold_values, mask in self.predict(args, lang, task):
                    self.accuracy_d[lang][task](pred_values, gold_values, mask)

                    
class CorrelationReporter(Reporter):
    
    def __init__(self, args, network, tasks, dataset, dataset_name):
        super().__init__(args, network, dataset, dataset_name)
    

        self._languages = args.languages
        self._tasks = tasks
        self.corr_d = defaultdict(dict)

    def write(self, args):
        for lang in self._languages:
            for task in self._tasks:
                prefix = '{}.{}.{}.corr.'.format(self.dataset_name, lang, task)
                if args.objects_only:
                    prefix += "objects_only."
                
                with open(os.path.join(args.out_dir, prefix + 'results'), 'w') as corr_f:
                    scorr, pcorr = self.corr_d[lang][task].result()
                    corr_f.write(f'spearman: {scorr}\n'
                                 f'pearson: {pcorr}\n')
    
    def compute(self, args):
        for lang in self._languages:
            for task in self._tasks:
                self.corr_d[lang][task] = Correlation()
                for pred_values, gold_values, mask in self.predict(args, lang, task):
                    self.corr_d[lang][task](pred_values, gold_values, mask)
