# multi-task-MXNet
**This is the implement of the multi-task image classificaton in MXNet** 

## What's multi-task image classification
For example, If you want to do 4 classes classification which include dog, apple, orange, bananer. On the one hand, you want your classifier to classify these four classes, on the other hand, you want your classifier to classify whether the input image is dog or not. Multi-task image classification can be used to solve this problem.

**This implement doesn't need recompile MXNet and is very convenient for you to use.**

Firstly, I assume that you have installed and used MXNet normally. 

Secondly, multi-task image classification is different from multi-label image classification both in data and model. **I assume you want to do a 4 classes classification and you want to have two tasks. I define task 1 as the original classification and task 2 as determine whether the label is bigger than 0**. 

Finally, do as follows:

## Data
Your .lst file may like this(take 4 classes as example):

|ID	|label   |      image_name|
|:------|:-------|:---------------| 
|5247	|0.000000|	image1.jpg|
|33986	|1.000000|	image2.jpg|
|39829	|2.000000|	image3.jpg|
|15647	|3.000000|	image4.jpg|
|10369	|1.000000|	image5.jpg|
|22408	|3.000000|	image6.jpg|
|2598	|2.000000|	image7.jpg|


There are two example of `train_data.lst` and `val_data.lst` in `/multi-task-MXNet/data_example/`

**In this implement, we only use .lst and raw image as the input instead of .rec file.**

## pretrained model
You can download pretrained model of ResNet50 from [Google Drive](https://drive.google.com/open?id=0ByXcv9gLjrVcVkQxMVAzcklQU00).
Put this `resnet-50-0000.params` in `pretrained_model/` file.

## Train
 `train_multitask.sh` is the train script for you to start fine-tune quickly. You should open this script and change some configurations such as: --epoch, --model, --batch-size, --num-classes, --data-train, --data-val, --image-train, --image-cal, --num-examples, --lr, --gpus, --num-epoch, --save-result, --save-name.


Then run: 
```
sh train_multitask.sh
```
