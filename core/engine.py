from tqdm import tqdm
import torch
import wandb
import os
import psutil
from cpuinfo import get_cpu_info

from mlxtend.plotting import plot_confusion_matrix
#from utils.plots import plot_confusion_matrix

from utils.metrics import ComputeMetrics
from utils.early_stopping import EarlyStopping
from utils.scheduler import get_scheduler
from utils.wandb_logging import wandb_log, wandb_log_scalar
from utils.tracker import PerformanceTracker
from utils.check_dir import count_folders_by_name
from utils.txt_messages import PIPE_TXT, WHITE_TXT, GOLDEN_TXT ,CLEAR_TXT
from utils.common import mkdir
from utils.val import val

from utils.humanize import human_readable_number


os.environ["WANDB_SILENT"]="true"

# https://towardsdatascience.com/optimize-pytorch-performance-for-speed-and-memory-efficiency-2022-84f453916ea6

# ❌ ⚠️ ✅

import logging

logger = logging.getLogger(__name__)
logger.handlers = []
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter(
    '%(name)s - %(levelname)s - %(message)s')) # %(asctime)s -
logger.addHandler(sh)


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad is False)
    total = trainable + non_trainable
    return total, trainable, non_trainable

def train(conf: dict, opt, hyps : dict, arch : dict, wandb_conf : dict, state : dict):

    # login to WandB.
    wandb.login(key=wandb_conf['key'])

    # merging the configuration dictionaries to log into wandb.
    wandb_config = conf | hyps | arch | wandb_conf | vars(opt) | state

    with wandb.init(project=wandb_conf['project'], config=wandb_config, 
                    name=wandb_conf['name'], mode=opt.mode, 
                    id = state['run_id'] if opt.resume else None, 
                    resume = opt.resume) as run:

        logger.debug(f'{wandb_conf["upload_weights"] is True}')

        # creating artifact to store weights.
        if wandb_conf['upload_weights'] is True:
            weights_artifact = wandb.Artifact('weights', type="weights")
            
        if wandb_conf['analytic_files'] is True:
            analytic_artifact = wandb.Artifact(f'analysis', type="files")
            
        # upload dataloader
        files_artifact = wandb.Artifact('files', type="files")
        # Add file
        files_artifact.add_file(os.path.join(os.getcwd(), 'core', 'criterion.py'))
        files_artifact.add_file(os.path.join(os.getcwd(), 'core', 'loader.py'))
        files_artifact.add_file(os.path.join(os.getcwd(), 'core', 'model.py'))
        files_artifact.add_file(os.path.join(os.getcwd(), 'core', 'optimizer.py'))
                                
        files_artifact.add_file(os.path.join(os.getcwd(), 'config', 'data.yaml'))
        files_artifact.add_file(os.path.join(os.getcwd(), 'config', 'architecture.yaml'))
        files_artifact.add_file(os.path.join(os.getcwd(), 'config', 'wandb.yaml'))
        files_artifact.add_file(os.path.join(os.getcwd(), 'config', opt.hyps + '.yaml'))
        # finalize artifact.
        run.log_artifact(files_artifact)

        # setting logger level.
        logger.setLevel(opt.logging)
        
        # benchmark to find the best algorithm to compute convolution.
        torch.backends.cudnn.benchmark = True

        model, criterion, optimizer, loader = conf['model'], conf['criterion'], conf['optimizer'], conf['train_loader']
        
        if opt.resume:
            logger.info('Loading model weights [TCC ONLY]')
            model.load_state_dict(torch.load('/home/vinicius-cin/torch_classification/runs/ViT-B/16_OpenAI_BEST_HYPER_66/weights/best.model'))
            logger.info('Weights loaded with success!')
        

        # * get run name.
        mkdir('runs')
        run_name = wandb_conf['name']
        run_name =  run_name + '_' + \
            str(count_folders_by_name(directory = 'runs', folder_name = run_name))

        # * on_train_setup

        # * get lr scheduler
        logger.info('Instantiating a lr scheduler')
        scheduler = get_scheduler(optimizer, hyps, verbose=True)

        # * instantiate EarlyStopping
        early_stopping = EarlyStopping(patience=opt.patience, delta=opt.delta, checkpoint_path='checkpoint.pt', 
                    is_maximize=True if hyps['better'] == 'max' else False, save_model=False, verbose=False)
        
        # * instantiate performance tracker to save best model and resume information.
        if opt.resume:
            # setting same run name as previous.
            run_name = state['run_name']
            # load tracker with resume state.
            tracker = PerformanceTracker(mode = hyps['better'], save_dir=os.path.join(os.getcwd(),'runs', run_name), 
                                         save = wandb_conf['upload_weights'] or opt.save_weights , state = state, verbose=False)
        else:
            tracker = PerformanceTracker(mode = hyps['better'], save_dir=os.path.join(os.getcwd(), 'runs', run_name), 
                                         save = wandb_conf['upload_weights'] or opt.save_weights, verbose=False)
        
        # * end on_train_setup

        # avoid division by zero.
        # delta = 1e-6

        # * device info.

        if opt.device == 'cuda':
            cuda_str = torch.cuda.get_device_name('cuda:0')
            print(PIPE_TXT + f'{WHITE_TXT} running on {cuda_str}')
        else:
            info = get_cpu_info()
            cpu_str = f"{info['brand_raw']} ({psutil.cpu_count(logical=False)}/{psutil.cpu_count()}) {info['hz_advertised_friendly']} "
            print(PIPE_TXT + f'{WHITE_TXT} running on {cpu_str}')

        # * training.
        print(PIPE_TXT + f'{WHITE_TXT} 🚀 Starting Training for {opt.epochs} epochs{CLEAR_TXT}\n')

        if tracker.last_epoch != 0:
            logger.warning(f'\nResuming training from epoch: {tracker.last_epoch}\n')
            
        # warmup info
        total, trainable, non_trainable = count_parameters(model)
        # logger.info(f'total : {human_readable_number(total)}')
        # logger.info(f'trainable : {human_readable_number(trainable)}')
        # logger.info(f'non_trainable : {human_readable_number(non_trainable)}')

        for epoch in range(tracker.last_epoch ,opt.epochs):  # opt.epochs
            
            # * warmup
            if opt.warmup_epochs > 0:
                
                # * end of warmup
                if epoch == opt.warmup_epochs:

                    logger.info('Unfreezing model.')
                    for param in model.parameters():
                        param.requires_grad = True
                    
                    optimizer.param_groups[0]['lr'] = float(hyps['max_lr'])
                    logger.info(f'fine-tune learning rate: {optimizer.param_groups[0]["lr"]}')
                    
                    total, trainable, non_trainable = count_parameters(model)
                    logger.info(f'total : {human_readable_number(total)}')
                    logger.info(f'trainable : {human_readable_number(trainable)}')
                    logger.info(f'non_trainable : {human_readable_number(non_trainable)}')
                
                elif epoch < opt.warmup_epochs:
                    logger.info(f'warming up {epoch+1}/{opt.warmup_epochs}')
                    optimizer.param_groups[0]['lr'] = opt.warmup_lr
                    logger.info(f'warmup learning rate: {optimizer.param_groups[0]["lr"]}')
                    if epoch == 0:
                        logger.info(f'total : {human_readable_number(total)}')
                        logger.info(f'trainable : {human_readable_number(trainable)}')
                        logger.info(f'non_trainable : {human_readable_number(non_trainable)}')
                else:
                    pass
                    
            # * on_train_epoch_start
            # start metrics.
            metrics = ComputeMetrics(out_features = arch['out_features'], 
                                     dataset_size = loader.__len__(),
                                     device=opt.device, task = arch['job_type'])
            # *

            # activating training layers.
            model.train()

            # set loss to zero to starting training.
            train_loss = []

            # creating progress bar.
            progress_bar = tqdm(
                loader, desc=f'Training', total=loader.__len__())
        
            # * batch loop (loop thorough all batches).
  
            for data in progress_bar:

                optimizer.zero_grad(set_to_none=True)

                # * on_train_batch_start
                # * 
                # loading one batch of data from dataloader.
                images, labels = data['images'].to(opt.device, non_blocking=True),\
                                 data['labels'].to(opt.device, non_blocking=True)

                # ! TODO normalize and cast to float32 only in GPU.
                #images, labels = Cast(images, labels)

                # sets the gradients of all its parameters (including parameters of submodules) to None.
                """
                Setting gradients to zeroes by model.zero_grad() or optimizer.zero_grad() would execute memset for all parameters and update gradients 
                with reading and writing operations. However, setting the gradients as None would not execute memset and would update gradients with 
                only writing operations. Therefore, setting gradients as None is faster.
                """
        
                # forward pass through network.
                outputs = model(images)

                # getting predictions of outputs (class index with higher probability).
                match arch['job_type']:
                    case 'multiclass':
                        predictions = torch.argmax(outputs, 1)
                    case 'multilabel':
                        # logits to probabilities.
                        probabilities = torch.sigmoid(outputs)
                        # probabilities to predictions (True or False)
                        predictions = torch.where(probabilities > 0.5, torch.ones_like(probabilities), torch.zeros_like(probabilities))
                    case _:
                        raise ValueError("Undefined job type")

                logger.debug(f'predictions dtype: {predictions.dtype}')
                
                logger.debug(f'labels: {labels}\n')
                logger.debug(f'outputs: {outputs}\n')
                logger.debug(f'predictions: {predictions}\n')

                logger.debug(
                    f'type for loss| dtype - outputs: {type(outputs)} {outputs.dtype} \
                      labels: {type(labels)} {labels.dtype}\n')

                # loss computation.
                logger.debug(f'labels size: {labels.size()}\n')
                logger.debug(f'outputs size: {outputs.size()}\n')

                loss = criterion(outputs, labels) # outputs

                # calculate loss.
                logger.debug(f'images.size(0) {images.size(0)}\n') # batch size (may be != of parameter batch_size)

                train_loss.append(loss.item())

                # backward pass (back-propagation).
                loss.backward()

                # update weights/parameters.
                optimizer.step()

                # * on_train_batch_end
                metrics.step(predictions = predictions, targets = labels.type(torch.int))
                # *

            # * end batch loop (end of an epoch).

            # * on_train_epoch_end

            logger.debug(' %s', metrics.matrix)

            logger.debug(f'train | len(loader) {len(loader)} loader.__len__()*loader.batch_size {loader.__len__()*loader.batch_size}')

            train_loss = torch.tensor(train_loss).mean().item()

            # @ Validation metrics.
            val_results = val(model, loader = conf['val_loader'], criterion = criterion, 
                              calc_loss = True, arch = arch, device=opt.device, metrics_conf = {'top_k': opt.top_k, 'num_samples': opt.num_samples})
            
            val_metrics, val_loss = val_results['metrics'], val_results['loss']

            # * lr scheduler step.
            scheduler.step(torch.mean(getattr(val_metrics, hyps['pursuit_metric'])).item()) # torch.mean(val_metrics.recall).item()

            logger.info('lr: %s', scheduler._last_lr[0])

            end_epoch_log = f'train precision: {torch.mean(metrics.precision).item():.4f}'\
                            f'| train recall: {torch.mean(metrics.recall).item():.4f}'\
                            f'| train loss: {train_loss:.4f}'
            
            end_epoch_log_val = f'val precision:   {torch.mean(val_metrics.precision).item():.4f}'\
                            f'| val recall:   {torch.mean(val_metrics.recall).item():.4f}'\
                            f'| val loss:   {val_loss:.4f}'\
                            
            end_epoch_other =   f'val p top {val_metrics.top_k:03d}:   {torch.mean(val_metrics.top_k_precision).item():.4f}'\
                                f'| mAP:          {torch.mean(val_metrics.ap).item():.4f}'\
                                f'| p@{val_metrics.num_samples:03d}:      {torch.mean(val_metrics.precision_at_k).item():.4f}'\
                                                 
                            #f'| P@{val_metrics.k:03d}:        {torch.mean(val_metrics.precision_k).item():.4f}|'\

            progress_bar.write(f'\n|epoch {epoch+1}/{opt.epochs}|')
            progress_bar.write(end_epoch_log)
            progress_bar.write(f'{end_epoch_log_val}')
            progress_bar.write(f'{end_epoch_other}\n')

            wandb_log(epoch = epoch, item = {'train loss': train_loss, 'val loss': val_loss, 'learning rate': scheduler._last_lr[0]}, 
                      metrics = {'train': metrics, 'val': val_metrics}, classes = wandb_conf['class_names'])
            
            # * performance tracker step.

            # hyps['pursuit_metric']: tracker and early stopping metric 
            # (i.e the metric we look for maximize or minimize.
            # (we want to stop the training and save the model when this metric is max or min).
            tracker.step(metric = torch.mean(getattr(val_metrics, hyps['pursuit_metric'])).item(), epoch = epoch, 
                         state_dict = model.state_dict())
            
            # * early stopping step.
            early_stopping(torch.mean(getattr(val_metrics, hyps['pursuit_metric'])).item(), model)

            if early_stopping.early_stop:
                print(PIPE_TXT + f'{GOLDEN_TXT} ⚠️ Early stopping after reaching tolerance of {opt.patience}{CLEAR_TXT}\n')
                #print(f'⚠️  Early stopping after reaching tolerance of {opt.patience}.')
                break

            # * end on_train_epoch_end

        # * on_training_end

        # * save last model and resume info.
        if wandb_conf['upload_weights'] or opt.save_weights is True:
            tracker.save_state_dict(state_dict = model.state_dict(), model_name = 'last')

            tracker.save_resume_info(info= {'run_id': run.id,
                                            'run_name': run_name,
                                            'best_epoch': tracker.best_epoch, 
                                            'last_epoch': tracker.last_epoch,
                                            'best_metric': tracker.best_metric,
                                            'actual_metric': tracker.actual_metric})
        
        # * saving model artifacts (upload to wandb).
        if wandb_conf['upload_weights'] is True:
            # upload best model.
            weights_artifact.add_file(os.path.join(tracker.weights_dir, 'best.model'))
            # upload last model.
            weights_artifact.add_file(os.path.join(tracker.weights_dir, 'last.model'))
            # upload resume info.
            weights_artifact.add_file(os.path.join(tracker.save_dir, 'resume_info.yaml'))

            # finalize artifact.
            wandb.log_artifact(weights_artifact)

        # * if analysis is necessary, loading best model.
        if (wandb_conf['analytic_files'] or wandb_conf['error_analysis'] is True) and opt.mode == 'online':

            logger.info(f'\nStarting evaluation.\n')

            # * load the best model for evaluation.
            try:
                best_weights = torch.load(os.path.join(
                    tracker.weights_dir, 'best.model'))
                model.load_state_dict(best_weights)
                print(
                    f'[info]: best.model state dict loaded with success! ✅')
            except Exception as e:
                print(
                    f'[info]: ❌ Error loading best.model state dict, skipping analytic_files and error_analysis')
                
                wandb_conf['analytic_files'] = False
                wandb_conf['error_analysis'] = False

                print(e)
                

            # if save error analysis artifacts, set to create_table to True.
            _ = True if wandb_conf['error_analysis'] else False

            # calculate metrics for the best model.
            val_results = val(model, loader = conf['val_loader'], criterion = criterion, 
                              device=opt.device, calc_loss = True, arch = arch,
                              ea_conf = {'create_table' : _, 'classes': wandb_conf['class_names'], 'fast_mode': opt.fast_mode, 'norm' : hyps['Normalize']},
                              metrics_conf = {'top_k': opt.top_k, 'num_samples': opt.num_samples})
            
            val_metrics = val_results['metrics']
            val_table = val_results['table']

            if _:
                analytic_artifact.add(val_table, "val_dataset_table")

            # * log best val metrics
            wandb_log_scalar(scalars={'BEST VAL PRECISION': val_metrics.precision,
                                      'BEST VAL RECALL': val_metrics.recall,
                                      'BEST VAL F1 SCORE': val_metrics.f1_score,
                                      'BEST VAL ACCURACY': val_metrics.accuracy,
                                      f'BEST VAL PRECISION@{val_metrics.num_samples}': val_metrics.precision_at_k,
                                      f'BEST VAL PRECISION TOP {val_metrics.top_k}': val_metrics.top_k_precision,
                                      'BEST VAL mAP': val_metrics.ap})
            
            if conf['test_loader']:
                test_results = val(model, loader = conf['test_loader'], criterion = criterion, 
                              device=opt.device, calc_loss = False, arch = arch,
                              ea_conf = {'create_table' : _, 'classes': wandb_conf['class_names'], 'fast_mode': opt.fast_mode, 'norm' : hyps['Normalize']},
                              metrics_conf = {'top_k': opt.top_k, 'num_samples': opt.num_samples})

                test_metrics = test_results['metrics']
                test_table = test_results['table']

                if _:
                    analytic_artifact.add(test_table, "test_dataset_table")

                # * log best test metrics
                wandb_log_scalar(scalars={'BEST TEST PRECISION': test_metrics.precision,
                                        'BEST TEST RECALL': test_metrics.recall,
                                        'BEST TEST F1 SCORE': test_metrics.f1_score,
                                        'BEST TEST ACCURACY': test_metrics.accuracy,
                                        f'BEST TEST PRECISION@{test_metrics.num_samples}': test_metrics.precision_at_k,
                                        f'BEST TEST PRECISION TOP {test_metrics.top_k}': test_metrics.top_k_precision,
                                        'BEST TEST mAP': test_metrics.ap})

        # * saving analytic artifacts.
        if wandb_conf['analytic_files'] is True and opt.mode == 'online':

            match arch['job_type']:

                case 'multiclass':
                    # create confusion matrix figure.
                    fig, ax = plot_confusion_matrix(conf_mat= val_metrics.matrix.cpu().numpy(),
                                                    class_names=list(wandb_conf['class_names'].values()),
                                                    colorbar=True,
                                                    figsize=(12, 12))
                    
                    # add title to the Figure.
                    fig.suptitle('Confusion Matrix (Validation)', fontsize=16)
                    
                    # save figure to wandb.
                    analytic_artifact.add(wandb.Image(fig), "val_confusion_matrix_best_model")
                    # save locally.
                    fig.savefig(os.path.join(tracker.save_dir,'val_confusion_matrix_best_model'))

                    if conf['test_loader']:

                        # create confusion matrix figure for test set.
                        fig, ax = plot_confusion_matrix(conf_mat= test_metrics.matrix.cpu().numpy(),
                                                        class_names=list(wandb_conf['class_names'].values()),
                                                        colorbar=True,
                                                        figsize=(12, 12))
                        
                        # add title to the Figure.
                        fig.suptitle('Confusion Matrix (Test)', fontsize=16)
                    
                        # save figure to wandb.
                        analytic_artifact.add(wandb.Image(fig), "test_confusion_matrix_best_model")
                        # save locally.
                        fig.savefig(os.path.join(tracker.save_dir,'test_confusion_matrix_best_model'))
                
                case _:
                    # it's redundant information to save a confusion matrix for multilabel classification.
                    pass

            # finalize artifact.
            run.log_artifact(analytic_artifact)

        logger.info(f"Training finished successfully ✅")

        run.finish()
        # *

    return model
