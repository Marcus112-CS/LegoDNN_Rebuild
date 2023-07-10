# LegoDNN-Rebuild  

### 1. Environment setup  

​    Attention: make sure you've pull this repo before run commands below

```shell
conda create -n legodnn python==3.6
conda activate legodnn
```

​	then enter root path of this repo

```shell
pip install --upgrade pip setuptools
pip install -r environments.txt
```

### 2. Train teacher model

​	model supported are listed below :

- vgg16
- mobilenetV2
- resnet18
- inceptionV3

​    dataset supported :

- cifar10
- cifar100 ( default )

```shell
cd cv_task/image_classification/cifar
```

​    train the teacher model to get model's checkpoint (which can instruct block train in LegoDNN)

​	**Attention: You must customize the `root_path` & `dataset_path`**, all your train set and report (each epoch's acc & loss) are saved in root_path

​	if there are ValueError (OOM) , you may need to decrease num of workers via **`--num_workers=7/6/5/4/3/2/1/0`** (num_workers=0 always fix the error)

```shell
# train vgg16 on cifar100 for example
python cv_task/image_classification/cifar/main.py --model=vgg16 --root_path=$TEACHER_MODEL_PATH$ --dataset_path=$DATASET_PATH$ --dataset=cifar100
```

​	if you'd like to set other argumets , check via **`python cv_task/image_classification/cifar/main.py --help`**

​	**what we need is the `.pth` file of the teacher model under `$TEACHER_MODEL_PATH$/$DATASET$/$MODEL$/YYYY-MM-DD/HH-MM-SS/.pth`**

​	e.g. `$TEACHER_MODEL_PATH$/cifar100/vgg16/2023-07-01/10-00-47/vgg16.pth`

​	You can check teacher model's acc in `$TEACHER_MODEL_PATH$/$DATASET$/$MODEL$/YYYY-MM-DD/HH-MM-SS/report.txt`

​	The time spent in this process is logged in **`time_log/teacher_model_train_time`**

### 3. LegoDNN experiments

​	demo of models are listed below:

- [vgg16](experiments/image_classification/vgg.py)
- [mobilenetV2](experiments/image_classification/mobilenetv2.py)
- [resnet18](experiments/image_classification/resnet18.py)
- [inceptionV3](experiments/image_classification/inceptionv3.py)

#### **3.1** customize basic settings

​	what you need to set is **`dataset_root_dir, checkpoint_path, num_workers`**

​	**`checkpoint_path`** is the **'.pth'** file path after teacher model training in last step

​	all the other settings are remained as default 

​	if there are ValueError (OOM) , you may need to decrease num of workers via reset **`num_workers`** or use GPU with higher memory

#### 3.2 confirm the dataset you use 

​	before run the program, make sure the checkpoint's dataset is suitable with **`dataset_name`** in demo

#### 3.3 time log

​	To mesure the time LegoDNN use , I've write an additional component **[time_logger.py](cv_task/log_on_time/time_logger.py)**

​	you can use it in other project as well.

​	usage:

```python
import cv_task.log_on_time.time_logger import time_logger

# set log_dir & log's tittle when init
time_logger_obj = time_logger(log_dir='./log', title='time log demo')   
# 1. start the timer
time_logger_obj.start()
"""
	program part 1
"""
# 2. lap (for several times)
time_logger_obj.lap(time_delta_name='step 1') # name segment of each part
"""
	program part 2
"""
# 3. end (must do at the end of timer)
time_logger_obj.end(time_delta_name='step 2') # name segment of the last part
```

​	Time log of LegoDNN's demo are saved under **`time_log/legodnn_execute_time`**

### 4. Check the result 

​	You can find the log & result under **`experiments/image_classification/log & experiments/image_classification/result `**

