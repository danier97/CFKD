import os
import sys
import torch
import argparse

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

import ltr.models.bbreg.atom as atom_models
from ltr.admin import loading

def main():
    parser = argparse.ArgumentParser(description='Generate success and precision plots')
    parser.add_argument('backbone', type=str)

    args = parser.parse_args()

    if args.backbone in ['teacher', 'resnet18', 'default']:
        model_name = 'atom_resnet18'
        net = atom_models.atom_resnet18(backbone_pretrained=False, cpu=True)
        path = '/content/pytracking/pytracking/networks/atom_default.pth'
        

    elif args.backbone in ['resnet18tiny', 'resnet18small', 'resnet18medium']:
        model_name = 'atom_'+args.backbone
        net_constructor = getattr(atom_models, model_name)
        net = net_constructor(backbone_pretrained=False, cpu=True)
        path = '/content/pytracking/pytracking/networks/cfkd_'+args.backbone+'.pth.tar'

    elif args.backbone in ['mobilenet', 'cfkd']:
        model_name = 'atom_mobilenetsmall'
        net = atom_models.atom_mobilenetsmall(backbone_pretrained=False, cpu=True)
        path = '/content/pytracking/pytracking/networks/atom_cfkd.pth.tar'
    
    else:
        print('wrong model name')
        return 
    
    net = loading.load_weights(net, path, strict=True)

    net_type = type(net).__name__
    state = {
        # 'epoch': self.epoch,
        # 'actor_type': actor_type,
        'net_type': net_type,
        'net': net.state_dict(),
        'net_info': getattr(net, 'info', None),
        'constructor': getattr(net, 'constructor', None)
        # 'optimizer': self.optimizer.state_dict(),
        # 'stats': self.stats,
        # 'settings': self.settings
    }

    tmp_name = '/content/pytracking/pytracking/networks/'+model_name+'_cpu.tmp'
    torch.save(state, tmp_name)
    os.rename(tmp_name, '/content/pytracking/pytracking/networks/'+model_name+'_cpu.pth.tar')


if __name__ == '__main__':
    main()