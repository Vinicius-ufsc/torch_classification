import torch
from clip import clip
from tqdm import tqdm

from core.clip_modeling.utils import torch_save, torch_load

import logging

logger = logging.getLogger(__name__)
logger.handlers = []
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter(
    '%(name)s - %(levelname)s - %(message)s')) # %(asctime)s - 
logger.addHandler(sh)

class ClassificationHead(torch.nn.Linear):

    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return torch_load(filename)

def get_zeroshot_classifier(clip_model, template, class_dict, device = 'cuda'):

    logit_scale = clip_model.logit_scale

    clip_model.eval()
    clip_model.to(device)

    logger.info('Getting zeroshot weights.')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(class_dict.values()):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = clip.tokenize(texts).to(device) # tokenize
            embeddings = clip_model.encode_text(texts) # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()
        
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return classification_head

class ImageEncoder(torch.nn.Module):
    def __init__(self, model, device = 'cuda', keep_lang=False):
        super().__init__()

        self.model = model
        self.device = device

        self.model, self.train_preprocess, self.val_preprocess = clip.load(
            self.model, self.device, jit=False)
        
        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image encoder from {filename}')
        return torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head, process_images=True, eval_preprocess=False, device='cuda'):
        super().__init__()
        self.image_encoder = image_encoder.to(device)
        self.classification_head = classification_head.to(device)
        self.process_images = process_images
        self.eval_preprocess = eval_preprocess
        self.device = device
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def forward(self, inputs):
        if self.eval_preprocess:
            inputs = self.image_encoder.val_preprocess(inputs).unsqueeze(0).to(self.device)
        if self.process_images:
            inputs = self.image_encoder(inputs)
        outputs = self.classification_head(inputs)
        return outputs

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return torch_load(filename)
