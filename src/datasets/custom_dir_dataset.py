import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from src.utils.io_utils import ROOT_PATH


class CustomDirDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    def __init__(
        self,
        data_dir: str,
        shuffle_index=False,
        target_sr=16000,
        instance_transforms=None,
    ):
        """
        Args:
            data_dir (str): path to custom dataset.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """
        self.target_sr = target_sr
        self.instance_transforms = instance_transforms

        assert data_dir is not None, "You should provide path to dir"

        self._data_dir = Path(data_dir).absolute().resolve()

        audio_dataset_path = self._data_dir / "audio"
        video_dataset_path = self._data_dir / "mouths"

        index = []

        for i, wav_path in enumerate((audio_dataset_path / "mix").iterdir()):
            item = dict()
            item["mix"] = wav_path
            file_1, file_2 = str(wav_path.stem).split("_")

            item["video_1"] = video_dataset_path / (file_1 + ".npz")
            item["video_2"] = video_dataset_path / (file_2 + ".npz")

            if (audio_dataset_path / "s1").exists():
                item["s1"] = audio_dataset_path / "s1" / wav_path.name
                item["s2"] = audio_dataset_path / "s2" / wav_path.name

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

        video_1 = self.load_video(item["video_1"])  # [1, time, W, H]
        video_2 = self.load_video(item["video_2"])  # [1, time, W, H]
        video = torch.stack([video_1, video_2], dim=1)  # [1, 2, time, W, H]

        if not (self._data_dir / "audio" / "s1").exists():
            return {"mixture": mix_wav, "wav_path": item["mix"], "video": video, "sample_rate": self.target_sr}

        s1_wav = self.load_audio(item["s1"])
        s2_wav = self.load_audio(item["s2"])

        if self.instance_transforms and "audio" in self.instance_transforms.keys():
            audio_transform = self.instance_transforms["audio"]
            mix_wav = audio_transform(mix_wav)
            s1_wav = audio_transform(s1_wav)
            s2_wav = audio_transform(s2_wav)

        if self.instance_transforms and "video" in self.instance_transforms.keys():
            video_transform = self.instance_transforms["video"]
            video = video_transform(video)

        sources = torch.stack([s1_wav, s2_wav], dim=1)  # [1, 2, time]

        return {
            "mixture": mix_wav,
            "wav_path": item["mix"],
            "sources": sources,
            "video": video,
            "sample_rate": self.target_sr,
        }

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(str(path))
        assert audio_tensor.size(0) == 1, "Audio should have one channel"

        target_sr = self.target_sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def load_video(self, path):
        video_array = np.load(path)["data"]
        assert len(video_array.shape) == 3, "Video should have one channel"

        return torch.from_numpy(video_array).unsqueeze(0).to(torch.float32)  # for consistency

    def _shuffle_index(self, index):
        random.seed(42)
        random.shuffle(index)

        return index
