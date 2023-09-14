from torchvision.models import get_model, list_models
from torchsummary import summary
import torch
from torch import nn

# num_classes, architecture

def make_model(architecture, out_features, weights="DEFAULT", device='cuda'):
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
