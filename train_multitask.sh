python train_multitask.py --epoch 0 --model pretrained_model/resnet-50 --batch-size 16 --num-classes 4 \
--data-train /your/path/to/train_data.lst --image-train /your/path/to/your/image \
--data-val /your/path/to/test_data.lst --image-val /your/path/to/your/image \
--num-examples 100000 --lr 0.001 --gpus 0 --num-epoch 10 --save-result /your/path/to/save/model/ --save-name resnet-50
