from torchvision.models import get_model, list_models
from torchsummary import summary
import torch
from torch import nn

from clip.clip import available_models as clip_available_models
from clip import clip
from core.clip_modeling.zeroshot import get_zeroshot_classifier
from core.clip_modeling.modeling import ImageEncoder, ImageClassifier

from templates.templates import Templates

import logging

logger = logging.getLogger(__name__)
logger.handlers = []
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter(
    '%(name)s - %(levelname)s - %(message)s')) # %(asctime)s - 
logger.addHandler(sh)

# num_classes, architecture

# TODO add possibility to load model from path 
def make_model(architecture, #  clip model path or model architecture name 
               out_features, 
               class_dict, # clip
               weights="DEFAULT", 
               template_name = 'simple_template', # clip
               freeze_encoder = False, # clip
               load_from_path = False,
               device='cuda'
               ):
    
    if load_from_path:
        model = torch.load(architecture).to(device)
        return model
    
    if architecture in list_models():
        model = get_model_from_torchvision(architecture = architecture, 
                                           out_features = out_features, 
                                           weights = weights, 
                                           device = device)
        logger.info(f"Using torchvision model: {architecture}")
    elif architecture in clip_available_models():

        if hasattr(Templates(), template_name):
            templates = getattr(Templates(), template_name)
        else:
            raise Exception(f"Template {template_name} not found.")


        model = get_clip_classifier_from_pretrained(clip_path = architecture, 
                                                    class_dict = class_dict, 
                                                    templates = templates, 
                                                    freeze_encoder = freeze_encoder, 
                                                    device = device)
        logger.info(f"Using clip model: {architecture}")
    else:
        raise Exception(f"architecture must be one of list_models - call" / 
                        "torchvision.models.list_models() to checkout available models or clip_available_models()" /
                         " to checkout clip models. {architecture} not found.")

    return model

def get_model_from_torchvision(architecture, 
               out_features, 
               weights="DEFAULT", 
               device='cuda'):
    """
    >Description:
        Instantiate a model using torchvision get_model,
        change the last layer to match num of output features.
    >Args:
        architecture {str}: name of the architecture.
        out_features {int}: num of classes.
        device {str}: 'cuda' to use GPU or 'cpu' to use CPU.
    """
    assert architecture in list_models(
    ), "architecture must be one of list_models - call torchvision.models.list_models() to checkout available models."

    model = get_model(architecture, weights=weights)

    # change last layer.
    if 'resnet' in architecture:
        model.fc = torch.nn.Linear(
            in_features=model.fc.in_features, out_features=out_features, bias=True)

    if 'efficientnet' in architecture:
        model.classifier[1] = nn.Linear(
            in_features=model.classifier[1].in_features, out_features=out_features)

    model = model.to(device)

    if False:
        summary(model, (3, 160, 160))

    return model

# ------------------------------------------------------------------------------

def get_clip_classifier_from_pretrained(clip_path, 
                                        class_dict, 
                                        templates,
                                        freeze_encoder = False,
                                        classification_head = None,
                                        keep_lang = False, 
                                        device = 'cuda'):

    # create a torch.nn.Module that combines CLIP image encoder with a linear classifier
    # for end-to-end image classification training and inference

    """
    >Description:
        Create a torch.nn.Module that combines CLIP image encoder with a linear classifier.
    >Args:
        clip_path {str} : path to clip model or name of the model.
        class_dict {dict} : {class_id : class_name}
        templates {str} : name of the template to use.
        freeze_encoder {bool} : freeze encoder weights.
        classification_head {torch.nn.Module} : classification head to use, if None, create a new one.
        keep_lang {bool} : keep text_encoder.
        device {str} : 'cuda' to use GPU or 'cpu' to use CPU. 
    """

    if classification_head is None:

        # class_dict = load_class_names(class_path)

        zero_shot, processor_train, processor_val = clip.load(name = clip_path, device = device, 
                                                            jit=False, is_train=False, pretrained=True)
        
        classification_head = get_zeroshot_classifier(clip_model = zero_shot, 
                            template = templates, 
                            class_dict = class_dict, 
                            device = device)

        del zero_shot
        del processor_train
        del processor_val

    else:
        pass

    image_encoder = ImageEncoder(model = clip_path, device = device, keep_lang=keep_lang)

    image_classifier = ImageClassifier(image_encoder = image_encoder, 
                                    classification_head = classification_head, 
                                    device = device,
                                    process_images=True,
                                    eval_preprocess=False)

    del image_encoder
    del classification_head

    if freeze_encoder:

        logger.info('Freezing encoder weights.')

        for param in image_classifier.parameters():
            param.requires_grad = False

        for param in image_classifier.classification_head.parameters():
            param.requires_grad = True

    image_classifier = image_classifier.to(device)

    return image_classifier
