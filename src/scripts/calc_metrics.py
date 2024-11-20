import warnings
from pathlib import Path

import hydra
import torchaudio
from hydra.utils import instantiate

from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="../configs", config_name="calc_metrics")
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
    s1 = Path(config.ground_truth_1).absolute().resolve()
    s2 = Path(config.ground_truth_2).absolute().resolve()

    for audio_path in s1.iterdir():
        gt = torchaudio.load(str(audio_path))
        preds = torchaudio.load(str(save_path / "s1" / str(gt.name)))

        for met in metrics["inference"]:
            metrics.update(met.name, met(preds=preds, sources=gt))

    for audio_path in s2.iterdir():
        gt = torchaudio.load(str(audio_path))
        preds = torchaudio.load(str(save_path / "s2" / str(gt.name)))

        for met in metrics["inference"]:
            metrics.update(met.name, met(preds=preds, sources=gt))

    logs = evaluation_metrics.result()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
