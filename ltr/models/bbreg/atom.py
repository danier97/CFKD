import torch.nn as nn
import ltr.models.backbone as backbones
import ltr.models.bbreg as bbmodels
from ltr import model_constructor
from collections import OrderedDict


class ATOMnet(nn.Module):
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
        super(ATOMnet, self).__init__()

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
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]), mode=mode)
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]), mode=mode)

        # train_feat_iou = [feat for feat in train_feat.values()]
        # test_feat_iou = [feat for feat in test_feat.values()]
        train_feat_iou = [train_feat[layer] for layer in self.bb_regressor_layer]
        test_feat_iou = [test_feat[layer] for layer in self.bb_regressor_layer]

        # Obtain iou prediction
        iou_pred = self.bb_regressor(train_feat_iou, test_feat_iou,
                                     train_bb.reshape(num_train_images, num_sequences, 4),
                                     test_proposals.reshape(num_train_images, num_sequences, -1, 4))

        if mode == 'train':
            return iou_pred, train_feat, test_feat
            
        return iou_pred

    def extract_backbone_features(self, im, layers=None, mode=None):
        if layers is None:
            layers = self.bb_regressor_layer
        if mode == 'train':
            layers = self.feature_extractor.output_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)



@model_constructor
def atom_resnet18(iou_input_dim=(256,256), iou_inter_dim=(256,256), backbone_pretrained=True, cpu=False):
    # backbone
    backbone_net = backbones.resnet18(output_layers=['conv1','layer1','layer2','layer3'],
                                      pretrained=backbone_pretrained)

    # Bounding box regressor
    iou_predictor = bbmodels.AtomIoUNet(pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim, cpu=cpu)

    # if training CPU version, then need to fine-tune regressor
    regressor_grad = True if cpu else False

    net = ATOMnet(feature_extractor=backbone_net, bb_regressor=iou_predictor, bb_regressor_layer=['layer2', 'layer3'],
                  extractor_grad=False, regressor_grad=regressor_grad)

    return net


@model_constructor
def atom_resnet50(iou_input_dim=(256,256), iou_inter_dim=(256,256), backbone_pretrained=True):
    # backbone
    backbone_net = backbones.resnet50(output_layers=['conv1','layer1','layer2','layer3'],
                                      pretrained=backbone_pretrained)

    # Bounding box regressor
    iou_predictor = bbmodels.AtomIoUNet(input_dim=(4*128,4*256), pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    net = ATOMnet(feature_extractor=backbone_net, bb_regressor=iou_predictor, bb_regressor_layer=['layer2', 'layer3'],
                  extractor_grad=False)

    return net


@model_constructor
def atom_resnet18medium(iou_input_dim=(128,128), iou_inter_dim=(128,128), backbone_pretrained=False, cpu=False):
    # backbone
    backbone_net = backbones.resnet18medium(output_layers=['conv1','layer1','layer2','layer3'],
                                            pretrained=backbone_pretrained, inplanes=32)

    # Bounding box regressor
    iou_predictor = bbmodels.AtomMediumIoUNet(input_dim=(64,128), 
                                              pred_input_dim=iou_input_dim, 
                                              pred_inter_dim=iou_inter_dim,
                                              cpu=cpu)

    extractor_grad = False if cpu else True

    net = ATOMnet(feature_extractor=backbone_net, bb_regressor=iou_predictor, bb_regressor_layer=['layer2', 'layer3'],
                  extractor_grad=extractor_grad, regressor_grad=True)

    return net


@model_constructor
def atom_resnet18small(iou_input_dim=(64,64), iou_inter_dim=(64,64), backbone_pretrained=False, cpu=False):
    # backbone
    backbone_net = backbones.resnet18small(output_layers=['conv1','layer1','layer2','layer3'],
                                           pretrained=backbone_pretrained, inplanes=16)

    # Bounding box regressor
    iou_predictor = bbmodels.AtomSmallIoUNet(input_dim=(32,64), 
                                             pred_input_dim=iou_input_dim, 
                                             pred_inter_dim=iou_inter_dim,
                                             cpu=cpu)

    extractor_grad = False if cpu else True

    net = ATOMnet(feature_extractor=backbone_net, bb_regressor=iou_predictor, bb_regressor_layer=['layer2', 'layer3'],
                  extractor_grad=extractor_grad, regressor_grad=True)

    return net


@model_constructor
def atom_resnet18tiny(iou_input_dim=(32,32), iou_inter_dim=(32,32), backbone_pretrained=False, cpu=False):
    # backbone
    backbone_net = backbones.resnet18tiny(output_layers=['conv1','layer1','layer2','layer3'],
                                          pretrained=backbone_pretrained, inplanes=8)

    # Bounding box regressor
    iou_predictor = bbmodels.AtomTinyIoUNet(input_dim=(16,32), 
                                            pred_input_dim=iou_input_dim, 
                                            pred_inter_dim=iou_inter_dim,
                                            cpu=cpu)
    
    # if training CPU version, only need to fine-tune regressor
    extractor_grad = False if cpu else True

    net = ATOMnet(feature_extractor=backbone_net, bb_regressor=iou_predictor, bb_regressor_layer=['layer2', 'layer3'],
                  extractor_grad=extractor_grad, regressor_grad=True)

    return net


@model_constructor
def atom_mobilenetsmall(backbone_pretrained=True, cpu=False):
    # backbone
    backbone_net = backbones.mobilenet_v2(output_layers=['conv1','layer1','layer2','layer3'],
                                          pretrained=backbone_pretrained)

    # Bounding box regressor
    iou_predictor = bbmodels.AtomSmallIoUNet(input_dim=(32,64), cpu=cpu)
    
    # if training CPU version, only need to fine-tune regressor
    extractor_grad = False if cpu else True
    extractor_grad = False if backbone_pretrained else extractor_grad

    net = ATOMnet(feature_extractor=backbone_net, bb_regressor=iou_predictor, bb_regressor_layer=['layer2', 'layer3'],
                  extractor_grad=extractor_grad, regressor_grad=True)

    return net


@model_constructor
def atom_mobilenet(backbone_pretrained=True, cpu=False):
    # backbone
    backbone_net = backbones.mobilenet_v2(output_layers=['conv1','layer1','layer2','layer3'],
                                          pretrained=backbone_pretrained)

    # Bounding box regressor
    iou_predictor = bbmodels.AtomSmallIoUNet(input_dim=(32,64), cpu=cpu)
    
    # if training CPU version, only need to fine-tune regressor
    extractor_grad = False if cpu else True
    extractor_grad = False if backbone_pretrained else extractor_grad

    net = ATOMnet(feature_extractor=backbone_net, bb_regressor=iou_predictor, bb_regressor_layer=['layer2', 'layer3'],
                  extractor_grad=extractor_grad, regressor_grad=True)

    return net