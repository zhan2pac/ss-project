# Audio-Visual Source Separation project

## Installation

1. Install dependencies

```bash
pip install -r ./requirements.txt
```

2. Download checkpoints and pre-trained model

```bash
python3 src/scripts/download_models.py
```

## Training

If you want to reproduce training process run following command.

```bash
pyhton3 train.py -cn=TRAIN_CONFIG
```

Where `TRAIN_CONFIG` is one of the train config in `src/configs` dir.

## Inference

If you want to inference model on custom dataset run following command.

```bash
python3 inference.py -cn=inference \
datasets.inference.data_dir=PATH_TO_CUSTOM_DATASET \
inferencer.save_path=SAVE_FOLDER \
inferencer.from_pretrained=PATH_TO_MODEL \
metrics.audio_only=IS_AUDIO_ONLY_MODEL
```

Note: `PATH_TO_CUSTOM_DATASET` should contain folders `audio` and `mouths`. Folder `audio`
should contain `mix` and may contain `s1` and `s2` folders for ground truth audios.
Predictions will be saved to `data/saved/SAVE_FOLDER`.

To calculate metrics run command.

```bash
python3 calc_metrics.py -cn=calc_metrics \
save_path=SAVE_FOLDER \
mixture=PATH_TO_MIX \
ground_truth_1=PATH_TO_S1 \
ground_truth_2=PATH_TO_S2 \
metrics.audio_only=IS_AUDIO_ONLY_MODEL
```

## Measure Resources Consumption

You can calculate MACs, inference time, number of parameters, etc by running the following command.

```bash
python3 measure_resources.py -cn=measure_resources \
model=MODEL_NAME \
datasets.inference.data_dir=PATH_TO_CUSTOM_DATASET \
inferencer.from_pretrained=PATH_TO_MODEL
```

You can view possible `MODEL_NAME` in `src/configs/model` folder.

## Credits

We use [Project Template](https://github.com/Blinorot/pytorch_project_template) for well-structured code.
And Lip-reading model from [repository](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks) to extract video features.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
