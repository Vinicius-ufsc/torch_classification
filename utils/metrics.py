
# * Confusion Matrix
# * accuracy
# * precision
# * recall
# * F1 score
# * AP

import torch
from torchmetrics import ConfusionMatrix, AveragePrecision, Precision

class ComputeMetrics():

    """
    >Description:
        Compute the following: accuracy, precision, recall,
        f1 score, AP, Confusion Matrix.
    """

    def __init__(self, out_features, dataset_size, task='multiclass', 
                 threshold = 0.5, normalize = None, device = 'cuda'):

        """
        run every new epoch (1 epoch = pass trough all data once).

        >Args:
            out_features {int}: number of classes.
            dataset_size {int}: total number of examples in the dataset. 
        """
        
        # numerical stability.
        self.delta = 1e-6

        self.dataset_size = dataset_size
        self.out_features = out_features 
        self.task = task
        self.threshold = threshold
        self.normalize = normalize
        self.device = device

        # dictionary to save outputs and targets.
        self.data = {'outputs': torch.tensor([]).to(self.device), 
                     'targets': torch.tensor([], dtype=torch.int32).to(self.device)}
        
        # instantiate confusion matrix.
        self.conf_matrix = ConfusionMatrix(task=self.task, threshold = self.threshold, 
                                    num_classes = self.out_features, num_labels = self.out_features,
                                    normalize = self.normalize).to(self.device)
    
        # for ap calculation.
        self.ap = None

        # for precision k calculation.
        self.precision_at_k = None
        self.num_samples = 0

        # top_k_precision
        self.top_k_precision = None
        self.top_k = 0

        # precision and recall for each class.
        # ex 3 classes: tensor([0.0000, 0.4444, 0.5000])
        self.matrix = None
        self.precision = 0
        self.recall = 0
        self.f1_score = 0
        self.accuracy = 0

    def step(self, predictions, targets, 
             outputs = None, save_data = False):
        """
        * run each bach.
        save_data: saves predictions and targets to compute AP.
        """
        # update all metrics.
        self.conf_matrix.update(predictions, targets)
        self.matrix = self.conf_matrix.compute()

        match self.task:
            case 'multiclass':
                self.precision = self.matrix.diag()/(self.matrix.sum(1) + self.delta)
                self.recall = self.matrix.diag()/(self.matrix.sum(0) + self.delta)
                self.f1_score = 2 * ((self.precision * self.recall) /
                                    (self.precision + self.recall + self.delta))

                # ! clarify acc = sum.
                self.accuracy =  self.matrix.diag() / self.matrix.sum()

            case 'multilabel':

                #true
                #    0 [TN FP]
                #    1 [FN TP]
                #        0  1
                #        pred

                # TP / (TP + FP) 
                self.precision = self.matrix[:,1,1] / (self.matrix[:,1,1] + self.matrix[:,0,1] + self.delta) 

                # TP / (TP + FN)
                self.recall = self.matrix[:,1,1] / (self.matrix[:,1,1] + self.matrix[:,1,0]) 

                self.f1_score = 2 * ((self.precision * self.recall) /
                                    (self.precision + self.recall + self.delta))
                
                # TP + TN / (TP + TN + FP + FN)
                self.accuracy = (self.matrix[:,1,1] + self.matrix[:,0,0]) \
                      / (self.matrix[:,1,1] + self.matrix[:,0,0] + self.matrix[:,1,0] + self.matrix[:,0,1]) 
                
        if save_data:

            # store the outputs and targets.
            self.data['outputs'] = torch.concat((self.data['outputs'],outputs))
            self.data['targets'] = torch.concat((self.data['targets'],targets))
                
    def compute_average_precision(self, dataset_predictions, dataset_targets):
        """
        > Description
            computes the average precision given a set of predictions and targets.
            AP-n: given all predictions and targets of class n, 
    
        > Args:
            dataset_predictions (torch.tensor): All the dataset predictions.
            dataset_targets (torch.tensor): All the dataset targets.
        """

        self._average_precision = AveragePrecision(num_labels=self.out_features, 
                                                    num_classes = self.out_features,
                                                    task=self.task, average=None).to(self.device)
        
        self.ap = self._average_precision(dataset_predictions, dataset_targets)


    def compute_precision_top_k(self, top_k ,dataset_predictions, dataset_targets):

        # ! TODO, implement for multiclass.
        if self.task == 'multilabel':

            self.top_k = top_k

            self._top_k_precision = Precision(top_k = self.top_k,
                                        num_labels=self.out_features, 
                                        num_classes = self.out_features,
                                        task=self.task, average=None,
                                        threshold = self.threshold).to(self.device)
        
            self.top_k_precision = self._top_k_precision(dataset_predictions, dataset_targets)
        
        else:
            self.top_k_precision = torch.tensor([0,0], dtype=torch.float)

    def compute_precision_at_k(self, num_samples ,dataset_predictions, dataset_targets):

        """
        -Para cada classe, ordenar as predições por confiança em ordem decrescente
        -Separar as K primeiras (K mais confiantes)

        Aplicar binarização -> independente do valor vai virar 1
        Comparar com as labels (com a mesma ordenação)
        Calculo da precisão
        """

        # ! TODO, implement for multiclass.

        if self.task == 'multiclass':
            # not implemented to multiclass.
            self.precision_at_k = torch.tensor([0,0], dtype=torch.float)
        else:

            self.num_samples = num_samples

            p_at_k_list = []

            # transpose.
            logits = torch.sigmoid(dataset_predictions)
            logits = torch.transpose(logits, dim0=0, dim1=1)
        
            targets = torch.transpose(dataset_targets, dim0=0, dim1=1)

            for p_class , t_class in zip(logits, targets):
                
                # sort
                sp_class, indices = torch.sort(p_class, descending=True)
                st_class = t_class[indices]

                # get num samples
                sp_class = sp_class[:self.num_samples]
                st_class = st_class[:self.num_samples]

                # apply threshold (the threshold is the lowest score, i.e all images are predicted as positive in the ith class).
                #sp_class = torch.where(sp_class > self.threshold, torch.ones_like(sp_class), torch.zeros_like(sp_class))
                sp_class = torch.ones_like(sp_class)

                mask = (sp_class == 1)
                mask_true = (mask & (st_class == 1))
                p_at_k = mask_true.sum() / (mask.sum() + self.delta)

                #p_at_k = torch.tensor([0]) if p_at_k.isnan().item() is True else p_at_k

                p_at_k_list.append(p_at_k)

            self.precision_at_k = torch.tensor(p_at_k_list, dtype=torch.float32)
