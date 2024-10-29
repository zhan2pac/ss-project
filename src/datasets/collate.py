import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    result_batch = {}

    result_batch["mixed"] = torch.cat([elem["mixed"] for elem in dataset_items], dim=0)  # [batch, time]
    result_batch["source"] = torch.cat([elem["source"] for elem in dataset_items], dim=0)  # [batch, 2, time]

    return result_batch
