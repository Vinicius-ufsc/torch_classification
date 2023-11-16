import logging
import torch
from torch.nn.functional import softmax
from utils.metrics import ComputeMetrics
import wandb
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.handlers = []
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter(
    '%(name)s - %(levelname)s - %(message)s')) # %(asctime)s -
logger.addHandler(sh)

def val(model, loader, arch, criterion = None, 
        device='cuda', calc_loss = True,
        ea_conf = {'create_table' : False, 'norm' : [[], [],  0]}, 
        metrics_conf = {'top_k': 3, 'num_samples': 25},
        verbose=False):
    
    """
    >Description:
        Evaluate the model.
    >Args:
        model: Model to evaluate
        loader: {DataloaderCsv} Dataloader
        arch: {dictionary}: must contain job_type and out_features (num of classes).
        criterion: {}: the criterion to calculate loss needed only if calc_loss = True.
        ea_conf {dictionary}: Table configuration for W&B artifact.
            norm: normalization hyperparameter from hyps.yaml, use the inverte the normalization.
        metrics_conf {dictionary}: Additional configuration for metrics.
        verbose {bool}: Whether to print logging messages or not. 
    >Returns: 
        {dictionary}: dictionary results with keys 'metrics', 'loss', 'table'.
    """

    if verbose:
        logger.info('Starting evaluation')

    if ea_conf['create_table']:

        # create wandb table and artifact.
        table = wandb.Table(
            columns=["file name", "image", "width", "height", "aspect ratio", 
                    "pixel resolution", "label", "prediction", "output_names", 
                    "output_softmax", "set"])
        
    # deactivate layers used only for training.
    model.eval()

    #with torch.inference_mode():
    with torch.no_grad():

        # set loss to zero.
        if calc_loss:
            loss_log = []

        metrics = ComputeMetrics(out_features = arch['out_features'], 
                                 dataset_size = loader.__len__(),
                                 device=device, task = arch['job_type'])

        # creating progress bar.
        if verbose:
            progress_bar = tqdm(
                loader, desc=f'Evaluating', total=len(loader))
        else:
            progress_bar = loader

        # looping through batches.
        for data in progress_bar:
            # loading one batch of data from dataloader.
            images, labels = data['images'].to(device, non_blocking=True),\
                             data['labels'].to(device, non_blocking=True)
            
            # forward pass.
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

            if ea_conf['create_table']:

                aspect_ratios = data['aspect_ratio']
                original_width, original_height  = data['original_width'], data['original_height']
                resolutions, names = data['resolution'], data['name']

                zipped_data = zip(  names, images, original_width, original_height,
                                    aspect_ratios, resolutions, labels, 
                                    predictions, outputs)
                
            # loss computation.
            if calc_loss:
                loss = criterion(outputs, labels) # outputs
                loss_log.append(loss.item())

            metrics.step(predictions = predictions, targets = labels.type(torch.int), outputs=outputs, save_data=True)

            if ea_conf['create_table']:

                table = create_error_analysis_table(table = table, zipped_data = zipped_data, task = arch['job_type'],
                                                    classes = ea_conf['classes'], fast_mode = ea_conf['fast_mode'], ea_conf = ea_conf)
            else:
                table = None

        # end of dataset forward.
        # compute AP.
        metrics.compute_average_precision(dataset_predictions = metrics.data['outputs'], 
                                          dataset_targets = metrics.data['targets'])
        
        # compute precision top_k.
        metrics.compute_precision_top_k(top_k = metrics_conf['top_k'], dataset_predictions = metrics.data['outputs'], 
                                    dataset_targets = metrics.data['targets'])
        
        # compute precision@K
        metrics.compute_precision_at_k(num_samples = metrics_conf['num_samples'], 
                                       dataset_predictions = metrics.data['outputs'], 
                                       dataset_targets = metrics.data['targets'])

        logger.debug(f'val | len(loader) {len(loader)} loader.__len__()*loader.batch_size {loader.__len__()*loader.batch_size}')
        loss = torch.tensor(loss_log).mean().item() if calc_loss else 0
        results = {'metrics': metrics, 'loss': loss, 'table': table}

        return results
    
import albumentations as A

def create_error_analysis_table(table, zipped_data, classes, ea_conf, task = 'multiclass', fast_mode = False):

    for name, image, original_width, original_height, aspect_ratio, resolution, label, prediction, output in zipped_data:
        
        # logger.critical(f'output size: {output.size()}')

        if task == 'multilabel':
            output = torch.sigmoid(output)
        elif task == 'multiclass':
            # ALTERADO PARA None
            output = softmax(output, dim=None)

        descending_args_id = list(torch.argsort(output))[::-1]

        descending_classes_names = [classes[i.item()]
                                    for i in descending_args_id]
        descending_probability = [
            round(output[i.item()].item(), 4) for i in descending_args_id]

        match task:
            case 'multiclass':
                prediction = classes[prediction.item()]
                label = classes[label.item()]
            case 'multilabel':

                prediction = [classes[i.item()] for i in torch.argwhere(prediction == 1)]
                prediction = ', '.join(prediction)

                label = [classes[i.item()] for i in torch.argwhere(label == 1)]
                label = ', '.join(label)
            case _:
                raise ValueError("Undefined task")
            

        image = image.cpu().numpy().transpose(1,2,0)

        # invert the normalization to saving the image.

        if ea_conf['norm'][-1] == 1 and not fast_mode:

            #inv_normalize = A.Normalize(
            #    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            #    std=[1/0.229, 1/0.224, 1/0.225], max_pixel_value=1
            #)

            inv_normalize = A.Normalize(
                mean=[-ea_conf['norm'][0][0]/ea_conf['norm'][1][0], \
                       -ea_conf['norm'][0][1]/ea_conf['norm'][1][1], \
                       -ea_conf['norm'][0][2]/ea_conf['norm'][1][2]], \
                std=[1/ea_conf['norm'][1][0], 1/ea_conf['norm'][1][1], \
                     1/ea_conf['norm'][1][2]], max_pixel_value=1
            )

            image = inv_normalize(image=image, bboxes=[(0.5, 0.5, 1, 1)], class_labels=[0])
            image = image['image']

        # possibility to not store images.
        img_file = 'no image' if fast_mode else wandb.Image(image)

        table.add_data(
            name,
            img_file,
            original_width,
            original_height,
            aspect_ratio,
            resolution,
            label,
            prediction,
            descending_classes_names,
            descending_probability,
            set)
        
    return table