import torch
import torchvision.models as models
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


def model_selV1(model_name):

    model = None

    if model_name == "vgg19":
        model = models.vgg19(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,2)

        param_names = [n for n, p in model.named_parameters()]
        for name in param_names[-10:]:
            param = dict(model.named_parameters())[name]
            param.requires_grad = True


    elif model_name == "efficientnet":
        model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=2)
        for param in model.parameters():
            param.requires_grad = False

        param_names = [n for n, p in model.named_parameters()]
        for name in param_names[-41:]:
            param = dict(model.named_parameters())[name]
            param.requires_grad = True


    elif model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)

        param_names = [n for n, p in model.named_parameters()]
        for name in param_names[-17:]:
            param = dict(model.named_parameters())[name]
            param.requires_grad = True



    else:
        print("check model")
        exit()


    return model


def model_selV2(model_name):

    model = None

    if model_name == "vgg19":
        model = models.vgg19(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.classifier[6].in_features

        model.classifier[6] = nn.Sequential(
                nn.Linear(num_ftrs, 2056),
                nn.BatchNorm1d(2056),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(2056, 1280),
                nn.BatchNorm1d(1280),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(1280, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256, 2) )


    elif model_name == "efficientnet":
        model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=2)
        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model._fc.in_features


        model._fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256, 2)
                )



    elif model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256,2 )
                )



    else:
        print("check model")
        exit()


    return model

def model_selV3(model_name):

    model = None

    if model_name == "vgg19":
        model = models.vgg19(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.classifier[6].in_features

        param_names = [n for n, p in model.named_parameters()]
        for name in param_names[-10:]:
            param = dict(model.named_parameters())[name]
            param.requires_grad = True

        model.classifier[6] = nn.Sequential(
                nn.Linear(num_ftrs, 2056),
                nn.BatchNorm1d(2056),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(2056, 1280),
                nn.BatchNorm1d(1280),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(1280, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256, 2) )


    elif model_name == "efficientnet":
        model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=2)
        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model._fc.in_features

        param_names = [n for n, p in model.named_parameters()]
        for name in param_names[-41:]:
            param = dict(model.named_parameters())[name]
            param.requires_grad = True

        model._fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256, 2)
                )

    elif model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.fc.in_features

        param_names = [n for n, p in model.named_parameters()]
        for name in param_names[-17:]:
            param = dict(model.named_parameters())[name]
            param.requires_grad = True
        model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256,2 )
                )
    else:
        print("check model")
        exit()


    return model

