from torchvision import datasets
import os
import torchvision.datasets as datasets

def download_voc(voc_root='./data'):
    """自动下载 VOC 2007 和 2012 数据集（检测+分割）"""
    os.makedirs(voc_root, exist_ok=True)

    print("正在下载 VOC 2007...")
    datasets.VOCSegmentation(root=voc_root, year='2007', image_set='test', download=True)
    datasets.VOCSegmentation(root=voc_root, year='2007', image_set='trainval', download=True)

    print("正在下载 VOC 2012...")
    datasets.VOCSegmentation(root=voc_root, year='2012', image_set='trainval', download=True)

    print("下载完成！数据保存在:", os.path.abspath(voc_root))

def prepare_project_structure():
    # 主工作目录
    work_dir = './voc_detection'
    os.makedirs(work_dir, exist_ok=True)
    
    # 模型目录
    model_dirs = ['mask_rcnn', 'sparse_rcnn']
    for model_dir in model_dirs:
        os.makedirs(os.path.join(work_dir, model_dir), exist_ok=True)
    
    # 可视化结果目录
    for model_dir in model_dirs:
        os.makedirs(os.path.join(work_dir, f'{model_dir}_visualizations'), exist_ok=True)
    
    print("项目目录结构创建完成!")

def main():
    """主函数"""
    download_voc()
    prepare_project_structure()

if __name__ == '__main__':
    main()