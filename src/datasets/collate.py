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

    result_batch["mixture"] = torch.cat([elem["mixture"] for elem in dataset_items], dim=0)  # [batch, time]
    result_batch["video"] = torch.cat([elem["video"] for elem in dataset_items], dim=0)  # [batch, 2, time, W, H]
    result_batch["sources"] = torch.cat([elem["sources"] for elem in dataset_items], dim=0)  # [batch, 2, time]
    result_batch["sample_rate"] = [elem["sample_rate"] for elem in dataset_items]

    return result_batch
