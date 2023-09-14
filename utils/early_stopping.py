import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, delta=0, checkpoint_path='checkpoint.pt', 
                 is_maximize=False, save_model=False, verbose=False):
        """
        >Args:
            patience {int} : How long to wait after last time validation loss improved.
                            Default: 10
            delta {float}: Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            checkpoint_path {str}: Path to save the model checkpoint.
                            Default: 'checkpoint.pt'
            is_maximize {bool}: Whether the metric to be monitored should be maximized or not.
                            Default: False
            save_model {bool}: Whether to save or not the model.
                            Default: False
            verbose {bool}: Whether to print or not info.
                            Default: False

        """
        self.patience = patience
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.is_maximize = is_maximize
        self.save_model = save_model
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = torch.tensor(float("Inf"))

    def __call__(self, val_loss, model):
        if self.is_maximize:
            score = val_loss
        else:
            score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if self.save_model:
                self.save_checkpoint(val_loss, model)

        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_model:
                self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.is_maximize:
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        else:
            if self.verbose:
                print(f'Validation loss increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.val_loss_min = val_loss