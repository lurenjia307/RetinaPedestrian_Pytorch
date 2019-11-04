

##### Data
1. Download CityPersons  VOCdevkit  dataset

2. convert  annotations as follow£º
   retina-train.txt
      content as£º

	# G:\datasets\pedestrian\CityPersons\images\train\aachen_000000_000019_leftImg8bit.png
	892 445 913 498  #lefttop point  right bottom point
	901 443 935 498
	1844 436 1888 542
	# G:\datasets\pedestrian\CityPersons\images\train\aachen_000002_000019_leftImg8bit.png
	1290 425 1315 486

## Train
```
$ train.py [-h] [data_path DATA_PATH] [--batch BATCH]
                [--epochs EPOCHS]
                [--shuffle SHUFFLE] [img_size IMG_SIZE]
                [--verbose VERBOSE] [--save_step SAVE_STEP]
                [--eval_step EVAL_STEP]
                [--save_path SAVE_PATH]
                [--depth DEPTH]
```

#### Example
For multi-gpus training, run:
```
$ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3 train.py 

#### Training log
```
---- [Epoch 39/200, Batch 400/403] ----
+----------------+-----------------------+
| loss name      | value                 |
+----------------+-----------------------+
| total_loss     | 0.09969855844974518   |
| classification | 0.09288528561592102   |
| bbox           | 0.0034053439740091562 |
| landmarks      | 0.003407923271879554  |
+----------------+-----------------------+
-------- RetinaFace Pytorch --------
Evaluating epoch 39
Recall: 0.7432201780921814
Precision: 0.906913273261629
```

##### Pretrained model

## Todo: 
- [ ] 