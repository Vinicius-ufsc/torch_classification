import argparse
import torch
import os

from distutils.util import strtobool
from templates.templates import Templates
from utils.common import mkdir
from pathlib import Path
import yaml

from core.model import get_clip_classifier_from_pretrained

"""
create and save a clip zero-shot classifier model.
"""

def parse_opt():

    parser = argparse.ArgumentParser()

    parser.add_argument('--clip_path', type=str, default='',
                        help='pretrained clip model path.')
    
    parser.add_argument('--name', type=str, default='zero_shot_model',
                    help='model name.')
    
    parser.add_argument('--yaml_path', type=str, default='zero_shot_template',
                        help='class yaml dictionary file path.')
    
    parser.add_argument('--templates', type=str, default='simple_template',
                        help='templates file path.')
    
    parser.add_argument('--freeze_encoder', type=lambda b:bool(strtobool(b)), nargs='?', const=False, default=False,
                            help='freeze encoder weights.')

    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda to run on GPU | cpu to run on CPU.')

    return parser.parse_args()

def main(opt, class_dict, templates):
    model = get_clip_classifier_from_pretrained(clip_path = opt.clip_path, 
                                                class_dict = class_dict, 
                                                templates = templates, 
                                                freeze_encoder = opt.freeze_encoder, 
                                                device = opt.device)

    save_path = os.path.join('zero_shot_models')
    mkdir(save_path)

    model_file = f"{opt.name}.pth"
    torch.save(model, os.path.join(save_path,model_file))
    print(f"Model saved with success | dir: {os.path.join(save_path,model_file)}")

if __name__ == "__main__":
    opt = parse_opt()

    with open(Path('config') / Path(opt.yaml_path + '.yaml'), "r") as _data:
        class_dict = yaml.load(_data, Loader=yaml.FullLoader)['names']
        _data.close()

    if hasattr(Templates(), opt.templates):
            templates = getattr(Templates(), opt.templates)
    else:
        raise Exception(f"Template {opt.templates} not found.")
    
    main(opt, class_dict, templates)
