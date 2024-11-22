import warnings
from pathlib import Path

import hydra
import torch
import torchaudio
from hydra.utils import instantiate

from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


def load_audio(path, target_sr):
    audio, sr = torchaudio.load(str(path))
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    return audio


@hydra.main(version_base=None, config_path="src/configs", config_name="calc_metrics")
def main(config):
    """
    Main script to calculate metrics.

    Args:
        config (DictConfig): hydra experiment config.
    """

    metrics = instantiate(config.metrics)

    evaluation_metrics = MetricTracker(
        *[m.name for m in metrics["inference"]],
        writer=None,
    )

    save_path = ROOT_PATH / "data" / "saved" / config.save_path

    mix = Path(config.mixture).absolute().resolve()
    s1 = Path(config.ground_truth_1).absolute().resolve()
    s2 = Path(config.ground_truth_2).absolute().resolve()

    for s1_path, s2_path in zip(s1.iterdir(), s2.iterdir()):
        mixture = load_audio(str(mix / s1_path.name), config.metrics.fs)
        gt1 = load_audio(str(s1_path), config.metrics.fs)
        gt2 = load_audio(str(s2_path), config.metrics.fs)

        preds1 = load_audio(str(save_path / "predicted_s1" / s1_path.name), config.metrics.fs)
        preds2 = load_audio(str(save_path / "predicted_s2" / s2_path.name), config.metrics.fs)

        for met in metrics["inference"]:
            evaluation_metrics.update(
                met.name,
                met(
                    preds=torch.stack([preds1, preds2], dim=1), sources=torch.stack([gt1, gt2], dim=1), mixture=mixture
                ),
            )

    logs = evaluation_metrics.result()

    for part in logs.keys():
        print(f"    {part:15s}: {logs[part].item()}")


if __name__ == "__main__":
    main()
