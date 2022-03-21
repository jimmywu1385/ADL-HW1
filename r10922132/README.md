# Homework 1 - Intent Classification & Slot Tagging

## Environment
same as environment provided by TA
## Preprocessing
download all preprocessing data
```
bash download.sh
```
## Train Intent
```
python train_intent.py --num_epoch 6 --hidden_size 756 --batch_size 64
```
## Train Slot
```
python train_slot.py --num_epoch 25 --rnn_type LSTM
```
## Test Intent
```
bash intent_cls.sh /path/to/test.json /path/to/pred.csv
```
## Train Slot
```
bash slot_cls.sh /path/to/test.json /path/to/pred.csv
```