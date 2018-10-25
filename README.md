## Introduction

Deep Learning models proven its power in many application areas in recent times, however its entry into embedded world 
as its own twists and practical difficulties. 

## Problem Statement

Come up with a framework that enables fast prototyping of Deep Learning models for Audio (to start with!) and provide a 
easy way to port the models on to Android using TFLite.  
 
## Proposed Solution

Come up with following modular components which can be then used as plug and play components:
 - Dataset modules with preprocessing modules
 - DataIterator modules
 - Tensorflow Models (Estimators)
 - Engine to run the models
 - Tensorflow model serving using TFLite
    - Web app
    - Mobile


## Dataset
- [FreeSound from Kaggle](https://www.kaggle.com/c/freesound-audio-tagging)
- Speech Recognition
    - [Kaggle](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge)
    - [Google](https://www.tensorflow.org/tutorials/sequences/audio_recognition)
    
TODO: add more details on how to download and for user data

-----------------------------------------------------------------------------------------------------------------------

## [Notebooks](notebooks)

-----------------------------------------------------------------------------------------------------------------------


## Python Environment

```
conda create -n shabda python=3.6 
source activate shabda
pip install tqdm
pip3 install tensorflow-gpu #system level installation
python3 -c "import tensorflow as tf; print(tf.__version__)"
pip install pandas
pip install numpy
pip install sklearn
pip install override


```

## Dataset
- https://www.kaggle.com/c/freesound-audio-tagging

```
cd data
./download.sh

```

## Commands:

**Train:**

```
python src/main/python/shabda/examples/cnn_naive_classifier/run.py --mode=train
```


**TF Lite**

```
cd /models/freesound/naive_conv_with_reshape/
mkdir android

MODEL_CHK_POINT=model.ckpt-876
INPUT_TENSOR=audio_features
OUTPUT_TENSOR=output-layer/softmax_output
OUTPUT_FROZEN_GRAPH=android/frozen_naive_conv.pb
OUTPUT_OPT_GRAPH=android/opt_naive_conv_graph.pb
TFLITE_FILE=android/graph.tflite

freeze_graph \
--input_graph=graph.pbtxt \
--input_checkpoint=$MODEL_CHK_POINT \
--input_binary=false \
--output_graph=$OUTPUT_FROZEN_GRAPH \
--output_node_names=$OUTPUT_TENSOR

python /home/mageswarand/anaconda3/envs/tensorflow1.0/lib/python3.6/site-packages/tensorflow/python/tools/optimize_for_inference.py \
--input=android/frozen_naive_conv.pb \
--output=$OUTPUT_OPT_GRAPH \
--frozen_graph=True \
--input_names=$INPUT_TENSOR \
--output_names=$OUTPUT_TENSOR

toco \
--graph_def_file=android/opt_naive_conv_graph.pb \
--input_format=TENSORFLOW_GRAPHDEF \
--output_format=TFLITE \
--inference_input_type=FLOAT \
--input_type=FLOAT \
--input_arrays=$INPUT_TENSOR \
--output_arrays=$OUTPUT_TENSOR \
--input_shapes=1,16384 \
--output_file=$TFLITE_FILE

```



