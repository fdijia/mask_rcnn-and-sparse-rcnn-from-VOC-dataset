# train mask_cnn，sparse_cnn on VOC2012

## 开始
-  定位工作目录并运行 `python prepareData.py`，该脚本将创建 `data/VOCdevkit/VOC2012/` 文件夹。
- 接着运行 `python voc2coco.py`（由于mmdet仅支持COCO格式的mask_cnn，需进行格式转换），脚本会生成 `data/coco` 文件夹。
- （建议直接下载COCO格式的VOC数据集，跳过上述两步操作）
- 运行 `python prepareModel.py`，程序将从./models目录读取脚本，在./voc_detection目录下创建 mask_cnn 和 sparse_cnn 模型结构（脚本源自mmdetection框架，可自行调整）。

- 运行 `python train.py` 启动训练。

- 训练完成后将获得日志文件和检查点，随后需运行 `weight_only.py` 将.pth文件转换为纯权重文件。同时可以运行 `toTensorBoard.py` 可视化训练过程。

- 然后可通过 `visualization_comparison.py` 使用生成的weight_only_file文件对模型进行可视化对比。


## start

- Locate the work directory and run `python prepareData.py` and it will create `data/VOCdevkit/VOC2012/` file.
- Then run `python voc2coco.py` (since mmdet only support coco format for mask_cnn and we should transform it) and it will create `data/coco` file.
- (We recommend you to download the VOC coco-formatted dataset directly and skip the two actions above).
- run `python prepareModel.py` and we will create `mask_cnn` and `sparse_cnn` in ./voc_detection from ./models (the scripts in the file are copied from mmdetection and fine-tuned, you can adjust it).
- run `python train.py` and it will start run.
- after train you will obtain log-file and checkpoint, then you should run `weight_only.py` to adjust the .pth file to weights only. And run `toTensorBoard.py` to visualize the train process
- then you can visualize your model by `visualization_comparison.py` through weight_only_file you got.

