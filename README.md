## <div align="center">Descriptions</div>

## YOLOv5 Improvements

In this project, we have made the following improvements to the official v6.1 version of YOLOv5:

1. **CBAM Attention Module**: We introduced the Channel Attention Module (CBAM) after the Convolutional (Conv) layer and the Spatial Pyramid Pooling (SPPF) layer. CBAM combines the attention mechanism of feature channels and feature space dimensions, enhancing the model's ability to focus on important regions and improve performance in object detection tasks.

2. **XML Label Generation**: We modified the detect code to include the `convert_to_voc_xml` function and the `parse_yolov5_predictions` function. These additions enable the use of the pre-trained model to generate VOC format XML label files. The generated XML files can be parsed through tools like labelimg, significantly reducing the cost of data labeling for object detection.

3. **Data Conversion Scripts**: We added a script to convert data in VOC format to YOLO format. This script allows seamless conversion between different data formats, making it easier to work with diverse datasets and train YOLOv5 models efficiently.

4. **Image File Integrity Check**: We also added a script to check whether the image files in the dataset are damaged. This check ensures that the input data is valid and avoids issues during training and evaluation.

These improvements contribute to the overall usability and performance of YOLOv5, making it a more versatile and effective tool for object detection tasks.
## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt]([https://github.com/ultralytics/yolov5/blob/master/requirements.txt](https://github.com/Joran1101/YOLOv5-CBAM-Seal-Detection/blob/main/requirements.txt)) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/Joran1101/YOLOv5-CBAM.git  # clone
cd YOLOv5-CBAM
pip install -r requirements.txt  # install
```
</details>

<details open>
<summary>Inference with detect.py</summary>

`detect.py` runs inference on a variety of sources, we have provided .pt model in "runs/train/exp26/weights" to detect seals in document images, can use following script to run the test example.
```bash
python detect.py --source ./seal-test --weights ./runs/train/exp26/weights/best.pt
```
BTW: When calling detect.py, if you do not comment out the convert_to_voc_xml function, parse_yolov5_predictions function, get_img_name function and the code line that quotes the error, you will get the VOC data format in the path of './data-gen/xml' xml file, the xml file can be normally parsed by data labeling tools such as labelimg, etc. You can use this technique to realize the function of using YOLOv5 for data pre-labeling, which can help you reduce the workload of data labeling.
     We have an example of using labelimg to parse xml files, see data-gen/labelImg.png

</details>

<details open>
<summary>training</summary>
The training command is as follows, it will use its own official seal dataset to train a model that can detect circular and oval official seals from pictures

```bash
python train.py --img 640 --batch 16 --epochs 300
```

`--img`: Image size for training (640x640).

`--batch`: Batch size for training (16).

`--epochs`: Number of training epochs (300).

`--dataset`: Path to the official seal dataset.

`--model`: Model configuration for oval seal detection.

<details open>

<summary>Exporting Static and Dynamic YOLOv5 Models to ONNX Format</summary>

To export the YOLOv5 model to ONNX format, you can use the following commands:

1. Exporting a Static Model:
```bash
python export.py --weights ./runs/train/exp26/weights/best.pt --include onnx
```

2. Exporting a Dynamic Model with Optimization:
```bash
python export.py --weights ./runs/train/exp26/weights/best.pt --include onnx --dynamic
```

- `--weights`: Specifies the path to the YOLOv5 model weights file (best.pt).
- `--include onnx`: Indicates that the exported model should be in ONNX format.
- `--dynamic`: If included, it enables dynamic ONNX export with optimizations.

The first command will export the YOLOv5 model from the `best.pt` file to an ONNX format file without dynamic optimization. The second command will export the YOLOv5 model with dynamic optimization enabled.
</details>

## <div align="center">Contact</div>

For YOLOv5 bugs and feature requests please visit [GitHub Issues](https://github.com/Joran1101/YOLOv5-CBAM-Seal-Detection/issues). 



