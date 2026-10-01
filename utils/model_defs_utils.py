from transformers import SwinModel
from torchvision import models
from torchvision.models import *
import timm
import torch.nn as nn
import torch

def load_resnet_34(hidden_layer):
    resnet_model = resnet34(pretrained=True)
    hidden_dim=None
    if hidden_layer is None:
        resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])
        hidden_dim = 512
    else:
        resnet_model = nn.Sequential(*list(resnet_model.children())[:4+hidden_layer])
    return resnet_model, hidden_dim

def load_resnet_50(hidden_layer):
    resnet_model = resnet50(pretrained=True)
    hidden_dim=None
    if hidden_layer is None:
        resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])
        hidden_dim = 2048
    else:
        resnet_model = nn.Sequential(*list(resnet_model.children())[:4+hidden_layer])
    return resnet_model, hidden_dim

def load_resnet_18(hidden_layer):
    resnet_model = resnet18(pretrained=True)
    hidden_dim=None
    if hidden_layer is None:
        resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])
        hidden_dim = 512
    else:
        resnet_model = nn.Sequential(*list(resnet_model.children())[:4+hidden_layer])
    return resnet_model, hidden_dim

def load_resnet_101(hidden_layer):  
    resnet_model = resnet101(pretrained=True)
    hidden_dim=None
    if hidden_layer is None:
        resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])
        hidden_dim = 2048
    else:
        resnet_model = nn.Sequential(*list(resnet_model.children())[:4+hidden_layer])
    return resnet_model, hidden_dim

def load_resnet_152(hidden_layer):
    resnet_model = resnet152(pretrained=True)
    hidden_dim=None
    if hidden_layer is None:
        resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])
        hidden_dim = 2048
    else:
        resnet_model = nn.Sequential(*list(resnet_model.children())[:4+hidden_layer])
    return resnet_model, hidden_dim

def load_swin_b(hidden_layer):
    model = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224")
    hidden_dim=None
    if hidden_layer is None:
        hidden_dim = 1024
    # print('bhojoms ?? ', model, hidden_dim)
    return model, hidden_dim

def load_swin_s(hidden_layer):
    model = SwinModel.from_pretrained("microsoft/swin-small-patch4-window7-224")
    hidden_dim=None
    if hidden_layer is None:
        hidden_dim = 768
    return model, hidden_dim

def load_swin_t(hidden_layer):
    model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    hidden_dim=None
    if hidden_layer is None:
        hidden_dim = 768
    return model, hidden_dim    

def load_swin_l(hidden_layer):
    model = SwinModel.from_pretrained("microsoft/swin-large-patch4-window7-224")
    hidden_dim=None
    if hidden_layer is None:
        hidden_dim = 1536
    return model, hidden_dim

def load_convnext_tiny(hidden_layer):
    model = convnext_tiny(pretrained=True)
    hidden_dim=None
    model = nn.Sequential(*list(model.children())[:-1])
    hidden_dim = 768
    return model, hidden_dim

def load_convnext_small(hidden_layer):
    model = convnext_small(pretrained=True)
    hidden_dim=Nonemodel = nn.Sequential(*list(model.children())[:-1])
    hidden_dim = 768
    return model, hidden_dim

def load_convnext_base(hidden_layer):
    model = convnext_base(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    hidden_dim = 1024
    return model, hidden_dim

def load_convnext_large(hidden_layer):
    model = convnext_large(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    hidden_dim = 1536
    return model, hidden_dim

def load_vgg11(hidden_layer):
    vgg11 = models.vgg11(pretrained=True)
    vgg11 = nn.Sequential(
        vgg11.features,
        vgg11.avgpool,
        nn.Flatten(),
        *list(vgg11.classifier.children())[:-1]  
    )
    hidden_dim=4096
    return vgg11, hidden_dim

def load_vgg13(hidden_layer):
    vgg13 = models.vgg13(pretrained=True)
    vgg13 = nn.Sequential(
        vgg13.features,
        vgg13.avgpool,
        nn.Flatten(),
        *list(vgg13.classifier.children())[:-1]  
    )
    hidden_dim=4096
    return vgg13, hidden_dim

def load_vgg16(hidden_layer):
    vgg16 = models.vgg16(pretrained=True)
    vgg16 = nn.Sequential(
        vgg16.features,
        vgg16.avgpool,
        nn.Flatten(),
        *list(vgg16.classifier.children())[:-1]  
    )
    hidden_dim=4096
    return vgg16, hidden_dim 

def load_vgg19(hidden_layer):
    vgg19 = models.vgg19(pretrained=True)
    vgg19 = nn.Sequential(
        vgg19.features,
        vgg19.avgpool,
        nn.Flatten(),
        *list(vgg19.classifier.children())[:-1]  
    )
    hidden_dim=4096
    return vgg19, hidden_dim

def load_vit_t(hidden_layer):
    vit_t_16 = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    hidden_dim=192
    return vit_t_16, hidden_dim

def load_vit_s(hidden_layer):
    vit_s_16 = timm.create_model('vit_small_patch16_224', pretrained=True)
    hidden_dim=384
    return vit_s_16, hidden_dim

def load_vit_b(hidden_layer):
    vit_b_16 = timm.create_model('vit_base_patch16_224', pretrained=True)
    hidden_dim=768
    return vit_b_16, hidden_dim

def load_vit_l(hidden_layer):
    vit_l_16 = timm.create_model('vit_large_patch16_224', pretrained=True)
    hidden_dim=1024
    return vit_l_16, hidden_dim

def load_dino_vit_s(hidden_layer):
    vit_s_16_dino = timm.create_model('vit_small_patch16_224.dino', pretrained=True)
    hidden_dim=384
    return vit_s_16_dino, hidden_dim

def load_dino_vit_b(hidden_layer):
    vit_b_16_dino = timm.create_model('vit_base_patch16_224.dino', pretrained=True)
    hidden_dim=768
    return vit_b_16_dino, hidden_dim

def load_dino_resnet50(hidden_layer):
    resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    hidden_dim=2048
    return resnet50, hidden_dim

def load_mocov2_resnet50(hidden_layer):
    resnet50 = models.resnet50(pretrained=False).cuda()     
    ckpt = torch.load('/home/khoslalab/Documents/universal embedding spaces/embeddings/moco_v2_800ep_pretrain.pth.tar')
    resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
    hidden_dim=2048
    return resnet50, hidden_dim


def load_vision_model(model_name, hidden_layer):
    if model_name == 'resnet50':
        return load_resnet_50(hidden_layer)
    elif model_name == 'resnet34':
        return load_resnet_34(hidden_layer)
    elif model_name == 'resnet18':
        return load_resnet_18(hidden_layer)
    elif model_name == 'resnet101': 
        return load_resnet_101(hidden_layer)
    elif model_name == 'resnet152':
        return load_resnet_152(hidden_layer)
    elif model_name == 'swinb':
        return load_swin_b(hidden_layer)
    elif model_name == 'swins':
        return load_swin_s(hidden_layer)
    elif model_name == 'swint':
        return load_swin_t(hidden_layer)    
    elif model_name == 'swinl':
        return load_swin_l(hidden_layer)
    elif model_name == 'convnext_tiny':
        return load_convnext_tiny(hidden_layer)
    elif model_name == 'convnext_small':        
        return load_convnext_small(hidden_layer)
    elif model_name == 'convnext_base':
        return load_convnext_base(hidden_layer)
    elif model_name == 'convnext_large':        
        return load_convnext_large(hidden_layer)
    elif model_name == 'vgg11':
        return load_vgg11(hidden_layer)
    elif model_name == 'vgg13':       
        return load_vgg13(hidden_layer)
    elif model_name == 'vgg16':
        return load_vgg16(hidden_layer)
    elif model_name == 'vgg19':
        return load_vgg19(hidden_layer)
    elif model_name == 'vit_t':
        return load_vit_t(hidden_layer)
    elif model_name == 'vit_s':
        return load_vit_s(hidden_layer)
    elif model_name == 'vit_b':       
        return load_vit_b(hidden_layer)
    elif model_name == 'vit_l':
        return load_vit_l(hidden_layer)
    elif model_name == 'dino_vit_s':
        return load_dino_vit_s(hidden_layer)
    elif model_name == 'dino_vit_b':    
        return load_dino_vit_b(hidden_layer)
    elif model_name == 'dino_resnet50':
        return load_dino_resnet50(hidden_layer)
    elif model_name == 'mocov2_resnet50':
        return load_mocov2_resnet50(hidden_layer)