

# Python Environment
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

# Dataset
- https://www.kaggle.com/c/freesound-audio-tagging

```
cd data
./download.sh

```

Commands:

**Train:**

```
python src/main/python/shabda/examples/cnn_naive_classifier/run.py --mode=train
```


**TF Lite**

```
cd /models/freesound/naive_conv_with_reshape/
mkdir android

INPUT_TENSOR=audio_features
OUTPUT_TENSOR=output-layer/softmax_output

freeze_graph \
--input_graph=graph.pbtxt \
--input_checkpoint=model.ckpt-59 \
--input_binary=false \
--output_graph=android/frozen_naive_conv.pb \
--output_node_names=$OUTPUT_TENSOR

python /home/mageswarand/anaconda3/lib/python3.6/site-packages/tensorflow/python/tools/optimize_for_inference.py \
--input=android/frozen_naive_conv.pb \
--output=android/opt_naive_conv_graph.pb \
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
--output_file=android/naive_conv.tflite

```

## Android
- https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/android
- https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android

### References:
- https://heartbeat.fritz.ai/intro-to-machine-learning-on-android-how-to-convert-a-custom-model-to-tensorflow-lite-e07d2d9d50e3
- https://sourcediving.com/machine-learning-on-mobile-fc34be69df1a
- https://blog.insightdatascience.com/ok-google-how-do-you-run-deep-learning-inference-on-android-using-tensorflow-c39fd00c427b **Java MFCC**
- https://www.simplifiedcoding.net/android-speech-to-text-tutorial/  **Android UI**
