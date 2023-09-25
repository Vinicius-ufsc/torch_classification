import torch
import os
import logging

logger = logging.getLogger(__name__)
logger.handlers = []
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)

class PerformanceTracker():

    """
    >Description:
        Keep track of model best metric value and best epoch,
        saves the model weights (state dict) every time the 
        metric was improved.
    """

    def __init__(self, mode = 'max', save = True, save_dir = '', verbose = False,
                 state : dict = {'actual_metric': 0, 'best_metric': -1, 'best_epoch': 0, 'last_epoch': 0}):

        self.mode = mode
        self.save = save
        self.save_dir = save_dir
        self.weights_dir =os.path.join(self.save_dir, 'weights') 
        self.verbose = verbose

        self.actual_metric = state['actual_metric']
        self.best_metric = state['best_metric']
        self.best_epoch = state['best_epoch']
        self.last_epoch = state['last_epoch']

        # create folders of not exist.
        self.create_folders()

        logger.setLevel(logging.INFO)

    def create_folders(self):
            # create the directory.
        try:
            os.makedirs(self.save_dir)
        except FileExistsError:
            # directory already exists
            pass
        try:
            os.makedirs(os.path.join(self.save_dir, 'weights'))
        except FileExistsError:
            # directory already exists
            pass
        
    def step(self, metric, epoch, state_dict):

        match self.mode:
            case 'max':
                if metric > self.best_metric:
                    self.best_metric = metric
                    self.best_epoch = epoch
                    if self.save:
                        self.save_state_dict(state_dict, model_name = 'best')
            case 'min':
                if metric < self.best_metric:
                    self.best_metric = metric
                    self.best_epoch = epoch
                    if self.save:
                        self.save_state_dict(state_dict, model_name = 'best')
            
        self.actual_metric = metric
        self.last_epoch = epoch

    def save_state_dict(self, state_dict, 
                        model_name = 'last', extension = '.model'):

        torch.save(state_dict, os.path.join(
                            self.save_dir, 'weights' ,model_name + extension))
        
        if self.verbose:
            logger.info(f'Model state dict saved with success | dir: {os.path.join(self.save_dir,model_name,extension)}')
        
    def save_resume_info(self, info : dict, filename = 'resume_info'):
        with open(os.path.join(self.save_dir, filename + '.yaml'), mode='w') as f:
            for key, value in info.items():
                f.write(f'{key}: {value}\n')
            f.close()

        if self.verbose:
            logger.info(f'resume info saved with success | dir: {os.path.join(self.save_dir,filename + ".yaml")}')

if __name__ == '__main__':
    tracker = PerformanceTracker(save_dir = 'weights', verbose=True)

    tracker.save_resume_info(info = {'a' : tracker.actual_metric, 'b': tracker.best_epoch})