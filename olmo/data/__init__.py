import json
from os.path import join
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from torch.utils.data import DataLoader, DistributedSampler

from ..aliases import PathOrStr
from ..config import DataConfig, TrainConfig, ObjectStoreConfig, VisionBackboneConfig
from ..exceptions import OlmoConfigurationError
from ..mm_data.data_iteration import IterationConfig
from ..mm_data.data_store import ExampleReader, MMStorageConfig
from ..mm_data.image_preprocessing import ImagePreprocessor, ResizeImage
from ..mm_data.iterable_dataset import MMIterableDataset
from ..mm_data.object_store import FileStore, ObjectStore
from ..torch_util import barrier, get_global_rank, get_world_size
from .collator import DataCollator
from .iterable_dataset import IterableDataset
from .memmap_dataset import MemMapDataset

__all__ = [
    "MemMapDataset",
    "DataCollator",
    "IterableDataset",
    "MMIterableDataset",
    "build_eval_dataloader",
    "build_train_dataloader",
]


def build_memmap_dataset(
    train_config: TrainConfig, data_config: DataConfig, include_instance_metadata: bool = True
) -> MemMapDataset:
    paths: List[str]
    metadata: List[Dict[str, Any]] = []
    if data_config.paths:
        if data_config.datasets:
            raise OlmoConfigurationError("DataConfig.paths is mutually exclusive with DataConfig.datasets")
        paths = data_config.paths
        for path in paths:
            metadata.append({"path": str(path)})
    elif data_config.datasets:
        paths = []
        for label in sorted(data_config.datasets.keys()):
            label_paths = data_config.datasets[label]
            paths.extend(label_paths)
            metadata.extend([{"label": label}] * len(label_paths))
    else:
        raise OlmoConfigurationError("One of DataConfig.paths or DataConfig.datasets is required")
    return MemMapDataset(
        *paths,
        chunk_size=train_config.model.max_sequence_length,
        metadata=metadata,
        include_instance_metadata=include_instance_metadata,
        pad_token_id=train_config.model.pad_token_id,
        generate_attention_mask=data_config.generate_attention_mask,
        label_mask_paths=cast(Optional[List[PathOrStr]], data_config.label_mask_paths),
    )


def build_eval_dataloader(
    train_config: TrainConfig,
    data_config: DataConfig,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = build_memmap_dataset(train_config, data_config, include_instance_metadata=True)
    collator = DataCollator(pad_direction=data_config.pad_direction, pad_token_id=train_config.model.pad_token_id)
    if data_config.drop_last:
        # Make sure batch size is small enough.
        samples_per_device = len(dataset) // get_world_size()
        batch_size = min(batch_size, samples_per_device)
        assert batch_size > 0, f"dataset for {data_config.paths} is too small"
    sampler = DistributedSampler(
        dataset,
        drop_last=data_config.drop_last,
        shuffle=shuffle,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        seed=train_config.seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=data_config.num_workers,
        sampler=sampler,
        pin_memory=data_config.pin_memory,
        prefetch_factor=None if data_config.num_workers == 0 else data_config.prefetch_factor,
        persistent_workers=False if data_config.num_workers == 0 else data_config.persistent_workers,
        timeout=data_config.timeout,
    )


def build_object_store(config: ObjectStoreConfig) -> ObjectStore:
    assert config.source_folder is not None
    return FileStore(config.source_folder)


def build_image_preprocessor(config: VisionBackboneConfig) -> ImagePreprocessor:
    return ResizeImage(
        (config.image_width, config.image_width),
        (config.patch_width, config.patch_height),
        config.resize_method
    )


def build_train_dataloader(train_config: TrainConfig) -> DataLoader:
    assert train_config.device_train_batch_size is not None
    collator = DataCollator(
        pad_direction=train_config.data.pad_direction, pad_token_id=train_config.model.pad_token_id
    )
    work_dir = Path(train_config.save_folder) / "train_data"
    if get_global_rank() == 0:
        if work_dir.is_dir() and not train_config.save_overwrite:
            raise OlmoConfigurationError(
                "train data working directory already exists, use --save_overwrite to overwrite"
            )
        else:
            work_dir.mkdir(exist_ok=True, parents=True)

    if train_config.data.multi_modal:
        data_cfg = train_config.data
        if train_config.model.vision_backbone is not None:
            vision_config = train_config.model.vision_backbone
            image_preprocessor = build_image_preprocessor(vision_config)
            object_store = build_object_store(data_cfg.object_store_config)
        else:
            object_store = None
            image_preprocessor = None
        it_config = IterationConfig(data_cfg.paths, data_cfg.sampler, data_cfg.sequence_builder)
        dataset = MMIterableDataset(
            data=it_config,
            object_store=object_store,
            image_preprocessor=image_preprocessor,
            idx_dir=data_cfg.idx_dir,
            seed=train_config.seed + (train_config.epoch or 0),
            sequence_length=train_config.model.max_sequence_length,
            global_batch_size=train_config.global_train_batch_size,
            drop_last=train_config.data.drop_last,
            num_threads=train_config.data.num_threads,
            thread_buffer_factor=train_config.data.thread_buffer_factor
        )
    else:
        dataset = IterableDataset(
            build_memmap_dataset(train_config, train_config.data, include_instance_metadata=False),  # type: ignore
            train_config.global_train_batch_size,
            seed=train_config.seed + (train_config.epoch or 0),
            shuffle=True,
            drop_last=train_config.data.drop_last,
            work_dir=work_dir,
            num_threads=train_config.data.num_threads
        )

    barrier()

    return DataLoader(
        dataset,
        batch_size=train_config.device_train_batch_size,
        drop_last=train_config.data.drop_last,
        collate_fn=collator,
        num_workers=train_config.data.num_workers,
        pin_memory=train_config.data.pin_memory,
        prefetch_factor=None if train_config.data.num_workers == 0 else train_config.data.prefetch_factor,
        persistent_workers=False if train_config.data.num_workers == 0 else train_config.data.persistent_workers,
        timeout=train_config.data.timeout,
    )
