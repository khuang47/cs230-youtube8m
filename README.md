#YouTube-8M understanding
Academic project for CS230 at Stanford University.

#Dataset
Please follow https://research.google.com/youtube8m/download.html to download both frame-level features dataset and
segment-rated frame-level features dataset.

##Video-level model
To train a model from start:
```
python train.py
```
The weights will be stored in experiments/base_model/weights.

To restore from saved checkpoints:
```
python train.py --model_dir experiments/base_model --restore_from experiments/base_model/weights
```

##Segment-level model (transfer learning from video-level model)
Copy the weights generated from the video-level model to experiments/segment_model/context_aware_weights and
experiments/segment_model/context_ignore_weights, then run:
```
python train_segment.py
```
The fully connected section of this model will be stored in experiments/segment_model/segment_weights.

To restore from saved checkpoints:
```
python train_segment.py --segment_restore_from experiments/segment_model/segment_weights
```
