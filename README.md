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

Where `TRAIN_CONFIG` is one of the following configs `train_rtfs_big`(final model), `train_artfs_big`, `train_sepreformer`.

## Inference

If you want to inference model on custom dataset run following command.

```bash
python3 inference.py -cn=INFERENCE_CONFIG \
datasets.inference.data_dir=PATH_TO_CUSTOM_DATASET \
inferencer.save_path=SAVE_FOLDER
```

Where `INFERENCE_CONFIG` is `inference_rtfs`(final model), `inference_artfs` or `inference_convtasnet`.

Note: `PATH_TO_CUSTOM_DATASET` should contain folders `audio` and `mouths`. Folder `audio`
should contain `mix` and may contain `s1` and `s2` folders for ground truth audios.
Predictions will be saved to `data/saved/SAVE_FOLDER`.

To calculate metrics run command.

```bash
python3 calc_metrics.py -cn=CALC_CONFIG \
save_path=SAVE_FOLDER \
mixture=PATH_TO_MIX \
ground_truth_1=PATH_TO_S1 \
ground_truth_2=PATH_TO_S2
```

Where `CALC_CONFIG` is `calc_metrics` if your predictions generated with AVSS model and `calc_metrics_audio`
if predictions generated with audio-only model.

## Measure Resources Consumption

You can calculate MACs, inference time, number of parameters, etc by running the following command.

```bash
python3 measure_resources.py -cn=MEASURE_CONFIG \
datasets.inference.data_dir=PATH_TO_CUSTOM_DATASET
```

Where `MEASURE_CONFIG` is `measure_rtfs`(final model), `measure_artfs` or `measure_convtasnet`.

## Credits

We use [Project Template](https://github.com/Blinorot/pytorch_project_template) for well-structured code.
And Lip-reading model from [repository](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks) to extract video features.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
