import random

import torch
import torchaudio
from torch.utils.data import Dataset

from src.utils.io_utils import ROOT_PATH


class MixDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    def __init__(
        self,
        part,
        data_dir=None,
        shuffle_index=False,
        target_sr=16000,
        instance_transforms=None,
    ):
        """
        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """
        self.part = part
        self.target_sr = target_sr
        self.instance_transforms = instance_transforms

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "dla_dataset" / "audio"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir

        dataset_path = self._data_dir / part

        index = []

        for i, wav_path in enumerate((dataset_path / "mix").iterdir()):
            item = dict()
            item["mix"] = wav_path
            if part != "test":
                item["s1"] = dataset_path / "s1" / wav_path.name
                item["s2"] = dataset_path / "s2" / wav_path.name

            index.append(item)

        if shuffle_index:
            index = self._shuffle_index(index)

        self._index = index

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        item = self._index[ind]

        mix_wav = self.load_audio(item["mix"])  # [1, time]

        if self.part == "test":
            return dict(mixed=mix_wav)

        s1_wav = self.load_audio(item["s1"])
        s2_wav = self.load_audio(item["s2"])

        if self.instance_transforms and "audio" in self.instance_transforms.keys():
            audio_transform = self.instance_transforms["audio"]
            mix_wav = audio_transform(mix_wav)
            s1_wav = audio_transform(s1_wav)
            s2_wav = audio_transform(s2_wav)

        source = torch.stack([s1_wav, s2_wav], dim=1)  # [1, 2, time]

        return dict(mixed=mix_wav, source=source)

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        assert audio_tensor.size(0) == 1, "Audio should have one channel"

        target_sr = self.target_sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def _shuffle_index(self, index):
        random.seed(42)
        random.shuffle(index)

        return index