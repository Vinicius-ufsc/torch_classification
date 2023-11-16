"""Optimizer for training"""

import torch

def make_optimizer(model, learning_rate=1e-3, optim='Adam', weight_decay=0.0005, momentum=0.937):
    """
    >Description:
        Instantiates an optimizer.
    >Args:
        model {torchvision.models}: model architecture.
        learning_rate {float}: learning rate.
        optim {str}: optimizer.
        weight_decay {float}: weight decay.
        momentum {float}: momentum.
    """

    assert optim in ['Adam', 'SGD', 'RMSprop', 'AdamW'], 'optimizers available: Adam, SGD, RMSprop'

    match optim:
        case 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                         weight_decay=weight_decay, betas=(momentum, 0.999))
        case 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                         weight_decay=weight_decay, betas=(momentum, 0.999))
        case 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                    momentum=momentum, dampening=0, weight_decay=weight_decay,
                                    nesterov=False, maximize=False,
                                    foreach=None)
        case 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate,
                                        alpha=0.99, eps=1e-08, weight_decay=weight_decay,
                                        momentum=momentum, centered=False, foreach=None)
        case _:
            raise ValueError("Undefined optimizer")

    return optimizer
