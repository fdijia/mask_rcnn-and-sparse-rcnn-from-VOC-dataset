import os
import numpy as np
import torch
import mmcv
import cv2
from mmengine.config import Config
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from mmengine.visualization import Visualizer
import torch
import matplotlib
import matplotlib.pyplot as plt

# 设置支持中文的字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# VOC类别
VOC_CLASSES = [
        'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'sofa', 'cow', 'dog',
        'horse', 'person', 'sheep', 'train', 'aeroplane',
        'chair', 'diningtable', 'motorbike', 'pottedplant', 'tvmonitor'
    ]


def get_proposal_boxes(model, img_path):
    """Get top-20 proposal boxes from Mask R-CNN's RPN"""
    img = mmcv.imread(img_path)
    img_shape = img.shape[:2]

    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    data_sample = DetDataSample()
    data_sample.set_metainfo(dict(
        img_shape=img_shape,
        scale_factor=np.array([1., 1.], dtype=np.float32),
        ori_shape=img_shape,
        batch_input_shape=img_shape
    ))
    data_sample.set_data(dict(
        gt_instances=InstanceData(),
        ignored_instances=InstanceData()
    ))

    data = {
        'inputs': [img_tensor.squeeze(0)],
        'data_samples': [data_sample]
    }

    with torch.no_grad():
        data = model.data_preprocessor(data, False)
        feats = model.extract_feat(data['inputs'])
        proposal_cfg = model.test_cfg.get('rpn', model.test_cfg)
        rpn_results_list = model.rpn_head.predict(
            feats,
            data['data_samples']
        )

    proposals = rpn_results_list[0].bboxes.cpu().numpy()
    return proposals[:20] if len(proposals) > 20 else proposals


def visualize_comparison(mask_rcnn_config, mask_rcnn_checkpoint,
                         sparse_rcnn_config, sparse_rcnn_checkpoint, test_images):

    save_dir = 'visualization/comparison'
    os.makedirs(save_dir, exist_ok=True)

    # 初始化模型
    mask_rcnn = init_detector(mask_rcnn_config, mask_rcnn_checkpoint, device='cpu')
    sparse_rcnn = init_detector(sparse_rcnn_config, sparse_rcnn_checkpoint, device='cpu')

    # 遍历测试图像
    for i, img_path in enumerate(test_images):
        print(f"处理图像 {i+1}/{len(test_images)}: {os.path.basename(img_path)}")
        img = mmcv.imread(img_path)
        img_rgb = mmcv.imconvert(img, 'bgr', 'rgb')
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # 1. Mask R-CNN Proposal Boxes
        proposals = get_proposal_boxes(mask_rcnn, img_path)
        proposal_img = img_rgb.copy()
        for box in proposals:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(proposal_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # 2. Inference
        mask_rcnn_result = inference_detector(mask_rcnn, img)
        sparse_rcnn_result = inference_detector(sparse_rcnn, img)

        # 子图布局
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(proposal_img)
        axes[0, 1].set_title(f'Mask R-CNN Proposals ({len(proposals)}个)')
        axes[0, 1].axis('off')

        # Mask R-CNN 结果
        mask_visualizer = VISUALIZERS.build(mask_rcnn.cfg.visualizer)
        mask_visualizer.dataset_meta = {'classes': VOC_CLASSES}

        mask_visualizer.add_datasample(
            'prediction',
            image=img_rgb.copy(),
            data_sample=mask_rcnn_result,
            draw_gt=False,
            draw_pred=True,
            show=False,
        )
        det_img = mask_visualizer.get_image()
        axes[1, 0].imshow(det_img)
        axes[1, 0].set_title('Mask R-CNN 结果')
        axes[1, 0].axis('off')

        # 目标检测 (Sparse R-CNN)
        sparse_visualizer = VISUALIZERS.build(sparse_rcnn.cfg.visualizer)
        sparse_visualizer.dataset_meta = {'classes': VOC_CLASSES}

        sparse_visualizer.add_datasample(
            'prediction',
            image=img_rgb.copy(),
            data_sample=sparse_rcnn_result,
            draw_gt=False,
            draw_pred=True,
            show=False,
        )
        det_img_sparse = sparse_visualizer.get_image()
        axes[1, 1].imshow(det_img_sparse)
        axes[1, 1].set_title('Sparse R-CNN 目标检测结果')
        axes[1, 1].axis('off')

        mask_count = len(mask_rcnn_result.pred_instances.bboxes)
        sparse_count = len(sparse_rcnn_result.pred_instances.bboxes)

        info_text = f"Mask R-CNN 检测到 {mask_count} 个物体，"
        info_text += f"Sparse R-CNN 检测到 {sparse_count} 个物体\n"

        if hasattr(mask_rcnn_result.pred_instances, 'scores') and len(mask_rcnn_result.pred_instances.scores) > 0:
            mask_conf = mask_rcnn_result.pred_instances.scores.mean().item()
            info_text += f"Mask R-CNN 平均置信度: {mask_conf:.3f}，"

        if hasattr(sparse_rcnn_result.pred_instances, 'scores') and len(sparse_rcnn_result.pred_instances.scores) > 0:
            sparse_conf = sparse_rcnn_result.pred_instances.scores.mean().item()
            info_text += f"Sparse R-CNN 平均置信度: {sparse_conf:.3f}\n"

        fig.suptitle(info_text, fontsize=16)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        save_path = os.path.join(save_dir, f'{img_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存对比图像到 {save_path}")

    print(f"所有可视化已保存至 {save_dir}")


def main():
    visualize_comparison(
        mask_rcnn_config='voc_detection/mask_rcnn_voc.py',
        mask_rcnn_checkpoint='voc_detection/mask_rcnn/epoch_12_weights_only.pth',
        sparse_rcnn_config='voc_detection/sparse_rcnn_voc.py',
        sparse_rcnn_checkpoint='voc_detection/sparse_rcnn/epoch_12_weights_only.pth',
        test_images=['data/coco/val2017/2010_004481.jpg', 'data/coco/val2017/2011_001110.jpg',
                     'data/coco/val2017/2011_003011.jpg', 'data/coco/val2017/2007_002234.jpg',
                     'data/img1.png', 'data/img2.png', 'data/img3.png']
    )

if __name__ == '__main__':
    main()