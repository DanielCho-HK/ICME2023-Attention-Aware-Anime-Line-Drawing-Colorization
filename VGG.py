import torch


def VGG(vgg_model, x):

    for param in vgg_model.parameters():
        param.requires_grad = False

    feature_maps = []
    f1 = torch.nn.Sequential(*list(vgg_model.features.children())[:4])(x)
    feature_maps.append(f1)
    f2 = torch.nn.Sequential(*list(vgg_model.features.children())[:9])(x)
    feature_maps.append(f2)
    f3 = torch.nn.Sequential(*list(vgg_model.features.children())[:18])(x)
    feature_maps.append(f3)
    f4 = torch.nn.Sequential(*list(vgg_model.features.children())[:27])(x)
    feature_maps.append(f4)
    f5 = torch.nn.Sequential(*list(vgg_model.features.children())[:36])(x)
    feature_maps.append(f5)

    return feature_maps



