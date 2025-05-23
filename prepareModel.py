import os
import mmengine
from mmengine import Config
from mmengine.runner import set_random_seed

def prepare_mask_rcnn_config(mask_rcnn_config_path):
    cfg = Config.fromfile('models/mask_rcnn.py')

    cfg.model.roi_head.bbox_head.num_classes = 20
    cfg.model.roi_head.mask_head.num_classes = 20
    cfg.work_dir = 'voc_detection/mask_rcnn'
    cfg.gpu_ids = range(1)
    cfg.optim_wrapper = dict(
        type='OptimWrapper',
        optimizer=dict(type='SGD', lr=0.01 / 8, momentum=0.9, weight_decay=0.0001)
    )
    cfg.seed = 49
    set_random_seed(49, deterministic=False)

    cfg.dump(mask_rcnn_config_path)
    return cfg, mask_rcnn_config_path

def prepare_sparse_rcnn_config(sparse_rcnn_config_path):
    cfg = Config.fromfile('models/sparse_rcnn.py')

    if hasattr(cfg.model, 'bbox_head'):
        cfg.model.bbox_head.num_classes = 20
    else:
        for i in range(len(cfg.model.roi_head.bbox_head)):
            cfg.model.roi_head.bbox_head[i].num_classes = 20
    cfg.work_dir = 'voc_detection/sparse_rcnn'
    cfg.gpu_ids = range(1)
    cfg.optim_wrapper = dict(
        optimizer=dict(
            type='AdamW', lr=0.000025, weight_decay=0.0001),
        clip_grad=dict(max_norm=1, norm_type=2))
    cfg.seed = 49
    set_random_seed(49, deterministic=False)

    cfg.dump(sparse_rcnn_config_path)
    return cfg, sparse_rcnn_config_path

def main():
    mask_rcnn_config_path = os.path.join('voc_detection', 'mask_rcnn_voc.py')
    prepare_mask_rcnn_config(mask_rcnn_config_path)
    sparse_rcnn_config_path = os.path.join('voc_detection', 'sparse_rcnn_voc.py')
    prepare_sparse_rcnn_config(sparse_rcnn_config_path)

if __name__ == '__main__':
    main()