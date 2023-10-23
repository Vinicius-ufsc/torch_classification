import pandas as pd
from collections import Counter
import torch

def return_weights(csv_file, job_type, device='cuda'):
        """
        return torch.tensor with balanced weights using the inverse frequency.
        """

        data = pd.read_csv(csv_file)

        match job_type:
            
                case 'multiclass':
                    
                        labels = data.iloc[:, 1].to_numpy()

                        counter = Counter(labels)
                        num_samples = sum(counter.values())
                        weights = num_samples / (torch.tensor(list(counter.values()), device=device, dtype=torch.float32))
                        print('weights:', weights)

                case 'multilabel':
                        data = data.iloc[: , 1:]

                        # Calcula a frequência de cada classe
                        positivas = data.sum(axis=0)
                        negativas = len(data)-positivas

                        weights = negativas / positivas
                        weights = torch.tensor(weights, device=device, dtype=torch.float32)
                        print('weights:', weights)
                
                case _:
                        raise ValueError("job_type not defined")

        return weights
