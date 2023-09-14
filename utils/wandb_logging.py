import wandb
import torch
from utils.metrics import ComputeMetrics

def wandb_log_scalar(scalars : dict):

    for key, value in scalars.items():
        wandb.log({f'{key}': torch.mean(value).item()})

def wandb_log(epoch : int, item : dict, metrics : dict[type[ComputeMetrics]], classes : dict):
        """
        wandb logging.
        """

        # choosing metrics to log.
        extract_mean = ['precision', 'recall', 'f1_score', 'accuracy', 'ap', 'top_k_precision', 'precision_at_k']
        get_value_by_class = ['precision', 'recall', 'f1_score', 'accuracy', 'ap', 'top_k_precision', 'precision_at_k']

        # loop through metrics (train metrics, val metrics).
        for _set, _class in metrics.items():
            # loop through atributes.

            # * logging mean values.
            for attribute in extract_mean:

                metric = getattr(_class, attribute)

                # to add K information.
                if attribute == 'top_k_precision': 
                     attribute = f'precision_top_{getattr(_class, "top_k")}'

                if attribute == 'precision_at_k':
                    attribute = f'precision_at_{getattr(_class, "num_samples")}'

                if metric is not None:
                    value = torch.mean(metric).item()
                else:
                     continue
                # log into wandb.
                wandb.log({f'{_set}_mean_{attribute}': value}, step=epoch)

            # * logging values by class.
            for attribute in get_value_by_class:
                metric = getattr(_class, attribute)

                # to add K information.
                if attribute == 'top_k_precision':
                    attribute = f'precision_top_{getattr(_class, "top_k")}'

                if attribute == 'precision_at_k':
                    attribute = f'precision_at_{getattr(_class, "num_samples")}'

                if metric is None:
                    continue
                # for each class in metric.
                for i, value in enumerate(metric):
                    wandb.log({f'{classes[i]}_{_set}_{attribute} - {i}': value}, step=epoch)

        # * itens, like losses.
        for key, value in item.items():
            wandb.log({f'{key}': value}, step=epoch)
