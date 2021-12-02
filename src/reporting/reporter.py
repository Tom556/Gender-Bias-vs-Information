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
        self.all_correct = 0
        self.all_predicted = 0

        self.male_correct= 0
        self.female_correct = 0
    
        self.all_male = 0
        self.all_female = 0
        self.threshold = threshold
        super().__init__()

    def __call__(self,gold, predicted, mask):
        self.update_state(gold.numpy(), predicted.numpy(), mask.numpy())
    
    def reset_state(self):
        self.all_correct = 0
        self.all_predicted = 0
        
        self.male_correct= 0
        self.female_correct = 0
        
        self.all_male = 0
        self.all_female = 0
    
    def update_state(self, predicted, gold, mask):
        self.all_predicted += np.sum(mask)
        correct = (gold == (predicted >= self.threshold))
        self.all_correct += np.sum(correct[mask.astype(bool)])
        
        self.male_correct += np.sum(correct[~gold.astype(bool) & mask.astype(bool)])
        self.female_correct += np.sum(correct[gold.astype(bool) & mask.astype(bool)])
        
        self.all_male += np.sum(mask[~gold.astype(bool)])
        self.all_female += np.sum(mask[gold.astype(bool)])

    def result(self):
        
        male_recall = self.male_correct / self.all_male
        female_recall = self.female_correct / self.all_female
        f1 = 2. * (male_recall * female_recall) / (male_recall + female_recall)
        if not self.all_predicted:
            return 0.
        return self.all_correct / self.all_predicted, male_recall, female_recall, f1
    

class Reporter():
    
    def __init__(self, args, network, dataset, dataset_name):
        self.network = network
        self.dataset = dataset
        self.dataset_name = dataset_name

    def predict(self, args, lang, task):
        data_pipe = Network.data_pipeline(self.dataset, [lang], [task], args, mode=self.dataset_name)
        progressbar = tqdm(enumerate(data_pipe), desc="Predicting, {}, {}".format(lang, task))
        for batch_idx, (_, _, batch) in progressbar:
            _, batch_bias, batch_information, batch_is_object, batch_embeddings = batch
            predicted = self.network.probe.predict_on_batch(batch_embeddings, lang, task)
            if 'bias' in task:
                gold, mask = self.network.probe._compute_target_mask(batch_bias)
            else:
                gold, mask = self.network.probe._compute_target_mask(batch_information)

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
                                     f'm_acc: {m_acc}\n'
                                     f'f_acc: {f_acc}\n'
                                     f'f1: {f1}\n')

    def compute(self, args):
        for lang in self._languages:
            for task in self._tasks:
                self.accuracy_d[lang][task] = Accuracy(self.threshold)
                for pred_values, gold_values, mask in self.predict(args, lang, task):
                    self.accuracy_d[lang][task](pred_values, gold_values, mask)
