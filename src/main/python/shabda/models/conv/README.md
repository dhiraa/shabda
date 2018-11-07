# Papers:
- http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

```commandline

cd /path/to/sarvam/src/

python speech_recognition/commands/run_experiments.py \
--mode=train \
--dataset-name=tensorflow_dataset_kaggle \
--data-iterator-name=raw_audio_data \
--model-name=simple_conv \
--batch-size=16 \
--num-epochs=1


python speech_recognition/commands/run_experiments.py \
--mode=train \
--dataset-name=speech_commands_v0 \
--data-iterator-name=audio_mfcc_google \
--model-name=cnn_trad_fpool3 \
--batch-size=32 \
--num-epochs=5

```
