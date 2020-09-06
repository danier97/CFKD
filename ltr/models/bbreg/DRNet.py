import torch.nn as nn
import ltr.models.backbone as backbones
import ltr.models.bbreg as bbmodels
from ltr import model_constructor
import torch

class DRNet(nn.Module):
    """ ATOM network module"""
    def __init__(self, feature_extractor, bb_regressor, bb_regressor_layer, extractor_grad=True, regressor_grad=True):
        """
        args:
            feature_extractor - backbone feature extractor
            bb_regressor - IoU prediction module
            bb_regressor_layer - List containing the name of the layers from feature_extractor, which are input to
                                    bb_regressor
            extractor_grad - Bool indicating whether backbone feature extractor requires gradients
        """
        super(DRNet, self).__init__()

        self.feature_extractor = feature_extractor
        self.bb_regressor = bb_regressor
        self.bb_regressor_layer = bb_regressor_layer

        if not extractor_grad:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)
        
        if not regressor_grad:
            for p in self.bb_regressor.parameters():
                p.requires_grad_(False)

    def forward(self, train_imgs, test_imgs, train_bb, test_proposals, mode=None):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        """
        num_sequences = train_imgs.shape[-4]
        num_train_images = train_imgs.shape[0] if train_imgs.dim() == 5 else 1
        num_test_images = test_imgs.shape[0] if test_imgs.dim() == 5 else 1

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.view(-1, *train_imgs.shape[-3:]), mode=mode)
        test_feat = self.extract_backbone_features(test_imgs.view(-1, *test_imgs.shape[-3:]), mode=mode)

        # For clarity, send the features to bb_regressor in sequence form, i.e. [sequence, batch, feature, row, col]
        train_feat_iou = [train_feat[layer].view(num_train_images, num_sequences, *train_feat[layer].shape[-3:])
                          for layer in self.bb_regressor_layer]
        test_feat_iou = [test_feat[layer].view(num_test_images, num_sequences, *test_feat[layer].shape[-3:])
                         for layer in self.bb_regressor_layer]

        # Obtain iou prediction
        reg_pred = self.bb_regressor(train_feat_iou, test_feat_iou,
                                     train_bb.view(num_train_images, num_sequences, 4),
                                     test_proposals.view(num_train_images, num_sequences, -1, 4))

        if mode == 'train':
            return reg_pred, train_feat, test_feat

        return reg_pred

    def extract_backbone_features(self, im, layers=None, mode=None):
        if layers is None:
            layers = self.bb_regressor_layer
        if mode == 'train':
            layers = self.feature_extractor.output_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)


@model_constructor
def drnet_resnet50(iou_input_dim=(512,1024), iou_inter_dim=(256,256), backbone_pretrained=True):
    # backbone
    backbone_net = backbones.resnet50(output_layers=['conv1','layer1','layer2','layer3'],
                                      pretrained=backbone_pretrained)

    # Bounding box regressor
    iou_predictor = bbmodels.DirectReg(input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    net = DRNet(feature_extractor=backbone_net, bb_regressor=iou_predictor, bb_regressor_layer=['layer2', 'layer3'],
                extractor_grad=False, regressor_grad=False)

    return net


@model_constructor
def drnet_se_resnet50(iou_input_dim=(512,1024), iou_inter_dim=(256,256), backbone_pretrained=True):
    # backbone
    backbone_net = backbones.se_resnet50(pretrained=backbone_pretrained)

    # Bounding box regressor
    iou_predictor = bbmodels.DirectReg(input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    net = DRNet(feature_extractor=backbone_net, bb_regressor=iou_predictor, bb_regressor_layer=['layer2', 'layer3'],
                  extractor_grad=False)

    return net


@model_constructor
def drnet_mobilenetsmall(backbone_pretrained=False):
    # backbone
    backbone_net = backbones.mobilenet_v2(output_layers=['conv1','layer1','layer2','layer3'],
                                          pretrained=backbone_pretrained)

    # Bounding box regressor
    iou_predictor = bbmodels.DirectRegSmall(input_dim=(32,64))

    net = DRNet(feature_extractor=backbone_net, bb_regressor=iou_predictor, bb_regressor_layer=['layer2', 'layer3'],
                extractor_grad=True, regressor_grad=True)

    return net
