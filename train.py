from core.loader import batch_loader, make_transformer
from core.model import make_model
from core.criterion import make_criterion
from core.optimizer import make_optimizer
from core.engine import train
from utils.balance_weights import return_weights
from clip.clip import available_models as clip_available_models
from utils.txt_messages import PIPE_TXT, WHITE_TXT, CLEAR_TXT

from distutils.util import strtobool

import yaml
import logging
from pathlib import Path
import argparse
import torch

from timeit import default_timer as timer
import datetime

logger = logging.getLogger(__name__)
logger.handlers = []
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter(
    '%(name)s - %(levelname)s - %(message)s')) # %(asctime)s - 
logger.addHandler(sh)

# csv_file, root_dir, transform=None, device = 'cuda',
# batch_size=16, num_workers=1, shuffle=False, generator = None, worker_init_fn = None

# architecture, out_features, weights="DEFAULT", device = 'cuda'

# job_type = 'Multiclass Classification', weight=None

# model, learning_rate=1e-3, optim='Adam', weight_decay=0.0005, momentum=0.937

# conf: dict, hyp, opt, device, callbacks
# model, criterion, optimizer, loader

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--logging', type=int, default=30, help='logging level | CRITICAL: 50, \
                         ERROR: 40, WARNING: 30, INFO: 20, DEBUG: 10, NOTSET: 0')

    parser.add_argument('--hyps', type=str, default='hyps_none',
                        help='')

    parser.add_argument('--arch', type=str, default='architecture',
                        help='')

    parser.add_argument('--data', type=str, default='data',
                        help='')

    # overridable arguments from hyps file.
    # ----------------------------------------------------------------
    parser.add_argument('--weight_decay', type=float, default=-1,
                        help='optimizer weight_decay, \
                        leave unchanged if weight_decay is set in the hyps file.')

    parser.add_argument('--momentum', type=float, default=-1,
                    help='optimizer momentum, \
                    leave unchanged if momentum is set in the hyps file.')

    parser.add_argument('--max_lr', type=float, default=-1,
                    help='optimizer max_lr, \
                    leave unchanged if max_lr is set in the hyps file.')

    parser.add_argument('--min_lr', type=float, default=-1,
                    help='optimizer min_lr, \
                    leave unchanged if min_lr is set in the hyps file.')

    # ----------------------------------------------------------------

    parser.add_argument('--mode', type=str, default='online',
                        help='set online for W&B logging or offline for local logging only.')
    
    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda to run on GPU | cpu to run on CPU.')
    
    parser.add_argument('--workers', type=int, default=0,
                        help='max dataloader workers')

    parser.add_argument('--epochs', type=int, default=3,
                        help='num of epochs to train.')

    parser.add_argument('--batch_size', type=int, default=2,
                        help='total batch size for all GPUs.')
    
    parser.add_argument('--patience', type=int, default=5,
                        help='how long to wait after last time validation metric improved.')
    
    parser.add_argument('--delta', type=int, default=0,
                        help='minimum change in the monitored quantity to qualify as an improvement (early stopping).')

    parser.add_argument('--balance_weights', type=lambda b:bool(strtobool(b)), nargs='?', const=False, default=False,
                        help='set custom weights for loss calculation based on class imbalance.')
    
    parser.add_argument('--fast_mode', type=lambda b:bool(strtobool(b)), nargs='?', const=False, default=False,
                        help='if True, does not upload dataset image files when creating error analysis table.')
    
    parser.add_argument('--resume', type=lambda b:bool(strtobool(b)), nargs='?', const=False, default=False,
                        help='resume last training.')
    
    parser.add_argument('--resume_info', type=str, default='',
                        help='resume training information.')
    
    parser.add_argument('--save_weights', type=lambda b:bool(strtobool(b)), nargs='?', const=False, default=False,
                        help='save weights locally.')
    
    parser.add_argument('--force_clip', type=lambda b:bool(strtobool(b)), nargs='?', const=False, default=False,
                        help='force clip pre-processor.')
        
    # ----------------------------------------------------------------

    parser.add_argument('--top_k', type=int, default=3,
                        help='k for top_k precision - ex: if 2, will use the top 2 confidence classes to compute confusion.')
    
    parser.add_argument('--num_samples', type=int, default=50,
                        help='num of samples for calculate precision@K.')

    return parser.parse_args()

class Pipeline():

    def __init__(self, opt, hyps, arch, data, wandb_conf, state):

        self.opt = opt
        self.hyps = hyps
        self.arch = arch
        self.wandb_conf = wandb_conf
        self.state = state

        self.train_data = data['train']
        self.val_data = data['val']
        self.test_data = data['test']
        self.root_dir = data['path']
        self.class_dict = data['names']

        self.architecture = arch['architecture']
        self.out_features = arch['out_features']
        self.job_type = arch['job_type']

        self.device = opt.device
        self.num_workers = opt.workers
        self.batch_size = opt.batch_size

        self.train_transform = make_transformer(
            aug_hyps=self.hyps, job_type='train')
        
        self.inference_transform = make_transformer(
            aug_hyps=self.hyps, job_type='inference')

        # if self.device == 'cuda':
        #     cuda_str = torch.cuda.get_device_name('cuda:0')
        #     logger.info(f"running on {cuda_str}")
        # else:
        #     info = get_cpu_info()
        #     cpu_str = f"{info['brand_raw']} ({psutil.cpu_count(logical=False)}/{psutil.cpu_count()}) {info['hz_advertised_friendly']} "
        #     logger.info(f"running on {cpu_str}")

    def fit(self):

        model = make_model(architecture=self.architecture,
                        out_features=self.out_features, 
                        weights=self.arch['weights'] if self.arch['weights'] != 'None' else None, 

                        template_name = self.arch['templates'],
                        freeze_encoder = self.arch['freeze_encoder'],
                        class_dict = self.class_dict,
                        device=self.device)
        
        # Check clip model
        # if self.opt.force_clip or self.architecture in clip_available_models():

        if self.opt.force_clip:
            logger.info("Using CLIP model pre-processors")
            is_clip = True
        else:
            is_clip = False

        if self.architecture in clip_available_models():
            logger.warning(f'⚠️ Image size must match {self.architecture} input size.')

        if is_clip:
            self.train_transform = model.val_preprocess if self.arch['freeze_encoder'] else model.train_preprocess
            self.inference_transform = model.val_preprocess

        train_data_loader = batch_loader(csv_file=self.train_data, root_dir=self.root_dir, 
                                        transform=self.train_transform, 
                                        job_type = self.arch['job_type'],
                                        batch_size=self.batch_size, 
                                        num_workers=self.num_workers, 
                                        shuffle=True, generator=None, worker_init_fn=None,
                                        is_clip = is_clip)
        
        val_data_loader = batch_loader(csv_file=self.val_data, root_dir=self.root_dir, 
                                        transform=self.inference_transform, 
                                        job_type = self.arch['job_type'],
                                        batch_size=self.batch_size, num_workers=self.num_workers, 
                                        shuffle=False, generator=None, worker_init_fn=None,
                                        is_clip = is_clip)
        
        if self.test_data is not None:
            test_data_loader = batch_loader(csv_file=self.test_data, root_dir=self.root_dir, transform=self.inference_transform, job_type = self.arch['job_type'],
                                         batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, generator=None, worker_init_fn=None, is_clip = is_clip)
        else:
            test_data_loader = None
        
        logger.info(f"Optim: {self.hyps['optim']}, lr: {float(self.hyps['max_lr'])}, decay: {float(self.hyps['weight_decay'])}, momentum: {float(self.hyps['momentum'])}")
        optimizer = make_optimizer(
            model=model, learning_rate=float(self.hyps['max_lr']), optim=self.hyps['optim'],
            weight_decay=float(self.hyps['weight_decay']), momentum=float(self.hyps['momentum']))

        # calculate weights for loss function.
        weight = None if self.opt.balance_weights is False else return_weights(csv_file=self.train_data,
                                                                          job_type = self.arch['job_type'], 
                                                                          device = self.device)
        criterion = make_criterion(
            job_type=self.job_type, weight=weight)

        model = train(conf={'model': model, 'criterion': criterion,
                    'optimizer': optimizer, 'train_loader': train_data_loader,
                    'val_loader': val_data_loader, 'test_loader' : test_data_loader}, 
                    opt=self.opt, arch=self.arch, hyps = self.hyps, wandb_conf=self.wandb_conf, 
                    state = self.state)

        return model

def main(opt):

    logger.setLevel(opt.logging)

    if opt.resume is True:
        assert opt.resume_info != '', 'To resume training, you must provide a resume_info.yaml file. Please check tracker at save_model.py.' 

    # hyps for training.
    with open(Path('config') / Path(opt.hyps + '.yaml'), "r") as hyps_file:
        hyps = yaml.load(hyps_file, Loader=yaml.FullLoader)
        hyps_file.close()

    # model architecture and loss function.
    with open(Path('config') / Path(opt.arch + '.yaml'), "r") as arch_file:
        arch = yaml.load(arch_file, Loader=yaml.FullLoader)
        arch_file.close()

    # path for dataset.
    with open(Path('config') / Path(opt.data + '.yaml'), "r") as _data:
        data = yaml.load(_data, Loader=yaml.FullLoader)
        _data.close()

    # path for dataset.
    with open(Path('config') / Path('wandb.yaml'), "r") as _wandb_conf:
        wandb_conf = yaml.load(_wandb_conf, Loader=yaml.FullLoader)
        _wandb_conf.close()

    # add class names into wandb_conf.
    wandb_conf['class_names'] = data['names']

    # path for dataset.
    if opt.resume:
        with open(Path(opt.resume_info), "r") as _state:
            state = yaml.load(_state, Loader=yaml.FullLoader)
            _state.close()
    else:
        state = dict()

    #
    #if opt.balance_weights == True:
    #    assert arch['job_type'] != 'multilabel', 'balance_weights Not implemented for multi-label classification, please, set it to False.'

    # print all parameters.
    if opt.logging == 10:
        print('\n|hyps|')
        for key, value in hyps.items() :
            print(f'{key} : {value}')
        
        print('\n|arch|')
        for key, value in arch.items() :
            print(f'{key} : {value}')

        #print('\n|data|')
        #for key, value in data.items() :
        #    print(f'{key} : {value}')

        #print('\n|W&B|')
        #for key, value in wandb_conf.items() :
        #    print(f'{key} : {value}')

        print('\n|opt|')
        for arg in vars(opt):
            print(f'{arg} : {getattr(opt, arg)}')
    
        if opt.resume:
            print('\n|state|')
            for key, value in state.items() :
                print(f'{key} : {value}')

        print('\n')

    # overwrite hyps if the argument is present in opt (cmd line argument).
    # this is necessary for better use of W&B sweep functionality.
    if opt.weight_decay != -1:
        logger.warning(f'⚠️ Overwriting weight_decay = {hyps["weight_decay"]} in {Path(opt.hyps + ".yaml")} file for {opt.weight_decay}')
        hyps['weight_decay'] = opt.weight_decay
    if opt.momentum != -1:
        logger.warning(f'⚠️ Overwriting momentum = {hyps["momentum"]} in {Path(opt.hyps + ".yaml")} file for {opt.momentum}')
        hyps['momentum'] = opt.momentum
    if opt.max_lr != -1:
        logger.warning(f'⚠️ Overwriting max_lr = {hyps["max_lr"]} in {Path(opt.hyps + ".yaml")} file for {opt.max_lr}')
        hyps['max_lr'] = opt.max_lr
    if opt.min_lr != -1:
        logger.warning(f'⚠️ Overwriting min_lr = {hyps["min_lr"]} in {Path(opt.hyps + ".yaml")} file for {opt.min_lr}')
        hyps['min_lr'] = opt.min_lr

    # check if hyps better parameter is valid.
    assert hyps['better'] in ['max', 'min'], 'unknown parameter.'

    agent = Pipeline(opt=opt, hyps=hyps, arch=arch, data=data, wandb_conf=wandb_conf, state = state)

    print(PIPE_TXT + f'{WHITE_TXT} 🤖 V-torch training pipeline{CLEAR_TXT}\n')

    start_time = timer()
    agent.fit()
    end_time = timer()

    elapsed_time = end_time - start_time

    print(PIPE_TXT + f'{WHITE_TXT} ⏰ Total training time using {opt.device} was {str(datetime.timedelta(seconds=elapsed_time))}{CLEAR_TXT}\n')
    #print(f"\n⏰ Total training time using {opt.device} was {str(datetime.timedelta(seconds=elapsed_time))}\n")

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
