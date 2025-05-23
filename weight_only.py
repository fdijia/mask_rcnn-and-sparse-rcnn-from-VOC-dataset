import torch
import os

def convert_to_weights_only(src_path, dst_path=None):
    checkpoint = torch.load(src_path, weights_only=False, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise ValueError('Checkpoint does not contain "state_dict".')

    if dst_path is None:
        base, _ = os.path.splitext(src_path)
        dst_path = base + '_weights_only.pth'

    torch.save(state_dict, dst_path)

if __name__ == '__main__':
    convert_to_weights_only('voc_detection/sparse_rcnn/epoch_12.pth')
