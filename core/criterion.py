from torch import nn

def make_criterion(job_type='Multiclass Classification', weight=None):
    """
    >Description:
        Returns a criterion based on the job_type.

        Binary Classification: 
            In binary classification, the goal is to classify an input into one of two possible classes. 
            For example, classifying an email as spam or not spam.

        Multiclass Classification: 
            In multiclass classification, the goal is to classify an input into one of several possible classes. 
            For example, classifying an image of a hand-written digit into one of ten possible digits.

        Multi-Label Classification:
            In multi-label classification, the goal is to assign one or more labels to an input. 
            For example, identifying the objects in an image and their locations.

        Hierarchical Classification:
            In hierarchical classification, the goal is to classify an input into one of several possible classes arranged in a hierarchy. 
            For example, classifying a bird image into its species and then its subspecies.

    >Args:
        job_type {str}: The type of classification problem.
        weight {torch.tensor}: The weights to the classes.
    """

    # one-hot encoded example:
    # logits = torch.tensor([[1.0, -2.0, 0.5], [2.0, 0.0, -1.0]])
    # targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

    # integer example:
    # input_tensor  = torch.tensor([1,2,5,6,7])
    # target_tensor  = torch.tensor([1,2,3,4,5])

    loss_functions = {'binary': nn.BCEWithLogitsLoss(pos_weight=weight),  # one-hot encoded.
                      # integer.
                      'multiclass': nn.CrossEntropyLoss(weight=weight),
                      'multilabel': nn.BCEWithLogitsLoss(pos_weight=weight)}  # one-hot encoded.

    assert job_type in loss_functions.keys(
    ), 'no supported criterion for the given job_type.'

    return loss_functions[job_type]

""" 
import torch

if __name__ == '__main__':
    loss = criterion(job_type = 'Multi-Label Classification', weight=None)

    logits = torch.tensor([[1.0, -2.0, 0.5], [2.0, 0.0, -1.0]])
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

    print(loss(logits, targets))

    loss = criterion(job_type = 'Multi-Label Classification', weight=torch.tensor([1.0,0.5,0.2]))

    print(loss(logits, targets))
"""
