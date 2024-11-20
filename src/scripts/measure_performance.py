import os
import time
import warnings

import hydra
import torch
from hydra.utils import instantiate
from thop import clever_format, profile

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="measure_performance")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    print(model)

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        metrics=None,
        skip_model_load=False,
    )

    for batch in dataloaders:
        batch = inferencer.move_batch_to_device(batch)
        batch = inferencer.transform_batch(batch)
        break

    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        _ = model(**batch)
        end = time.time()
        print("One batch inference time:", end - start, "s")
        print("Memory required to inference one batch:", torch.cuda.max_memory_allocated(), "bytes")
        torch.cuda.reset_peak_memory_stats()

    macs, params = profile(model, inputs=(batch,))
    macs, params = clever_format([macs, params], "%.3f")

    print("MACs:", macs)
    print("Number of parameters:", params)
    print("Size of model:", clever_format(os.path.getsize(config.inferencer.get("from_pretrained"))))


if __name__ == "__main__":
    main()
