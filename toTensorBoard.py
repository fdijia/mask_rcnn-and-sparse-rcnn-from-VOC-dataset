from torch.utils.tensorboard import SummaryWriter
import json
import glob
import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def is_evaluation_log(log_entry):
    """判断是否是评估日志"""
    return any(key.endswith('mAP') for key in log_entry.keys())

def toLog(base_dir):
    # 创建对应的 TensorBoard writer
    train_writer = SummaryWriter(os.path.join(base_dir, 'log_file/train'))
    val_writer = SummaryWriter(os.path.join(base_dir, 'log_file/val'))

    # 查找所有的 .json 文件
    json_files = glob.glob(os.path.join(base_dir, '*.json'))
    if not json_files:
        print("No JSON files found.")
        return

    json_file = json_files[0]
    print(f"Processing file: {json_file}")

    with open(json_file, 'r') as f:
        for line in f:
            try:
                j = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            if is_evaluation_log(j):
                writer = val_writer
                global_step = j.get('step')
                prefix = ''
            else:
                writer = train_writer
                global_step = j.get('step', j.get('iter'))
                prefix = ''

            for key, value in j.items():
                if isinstance(value, (int, float)):
                    # 移除coco/前缀，因为TensorBoard会自动创建层级
                    tb_key = key.replace('coco/', '')
                    writer.add_scalar(prefix + tb_key, value, global_step=global_step)

    train_writer.close()
    val_writer.close()
    print("TensorBoard logs written successfully.")

def draw(ea, name, save_path, save_name):
    data = ea.Scalars(name)
    steps = [x.step for x in data]
    values = [x.value for x in data]
    plt.figure(figsize=(10, 6))
    plt.plot(steps, values)
    plt.xlabel('Step')
    plt.ylabel(name)
    plt.savefig(os.path.join(save_path, save_name), dpi=300)
    plt.close()

def draw_mask(log_file, save_path):
    train_ea = event_accumulator.EventAccumulator(os.path.join(log_file, 'train'))
    train_ea.Reload()
    names = ['loss']
    for name in names:
        draw(train_ea, name, save_path, save_name=name+'.png')

    val_ea = event_accumulator.EventAccumulator(os.path.join(log_file, 'val'))
    val_ea.Reload()
    for name in val_ea.Tags()["scalars"]:
        draw(val_ea, name, save_path, save_name=name+'.png')


if __name__ == '__main__':
    base_dir = 'voc_detection/mask_rcnn/20250521_133642/vis_data'
    toLog(base_dir)
    mask_log = "voc_detection/mask_rcnn/20250521_133642/vis_data/log_file"
    draw_mask(mask_log, save_path="voc_detection/mask_rcnn_visualizations")

    base_dir = 'voc_detection/sparse_rcnn/20250523_063057/vis_data'
    toLog(base_dir)
    mask_log = "voc_detection/sparse_rcnn/20250523_063057/vis_data/log_file"
    draw_mask(mask_log, save_path="voc_detection/sparse_rcnn_visualizations")