import pandas as pd
from collections import Counter
import torch

def return_weights(csv_file, job_type, device='cuda'):
        """
        return torch.tensor with balanced weights for train dataset.
        """

        data = pd.read_csv(csv_file)

        match job_type:
                case 'multiclass':
                        # ! TODO corrigir para o caso multiclass.
                        labels = data.iloc[:, 1].to_numpy()

                        counter = Counter(labels)
                        num_samples = sum(counter.values())
                        weights = 1 - (torch.tensor(list(counter.values()), device=device, dtype=torch.float32) / num_samples)

                case 'multilabel':
                        data = data.iloc[: , 1:]

                        # Calcula a frequência de cada classe
                        positivas = data.sum(axis=0)
                        negativas = len(data)-positivas

                        weights = negativas / positivas
                        weights = torch.tensor(weights, device=device, dtype=torch.float32)
                
                case _:
                        raise ValueError("job_type not defined")

        return weights
