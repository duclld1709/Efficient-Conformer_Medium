"""Offline feature extraction for the ViMD dataset.

This script loads raw ViMD audio samples, runs the same audio preprocessing
module used in the Efficient Conformer encoder, and stores the resulting
spectrograms alongside tokenized transcripts. The cached features can later be
consumed via CachedFeatureDataset without keeping the original waveform
dataset on disk.
"""

import argparse
import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import sentencepiece as spm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.modules import AudioPreprocessing
from utils.datasets import ViMDDataset
from utils.preprocessing import collate_fn_pad


def parse_args():
    parser = argparse.ArgumentParser(description="Extract spectrogram features for ViMD")
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/EfficientConformerCTCSmall.json",
        help="Path to model config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store cached features",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test", "valid"],
        help="Dataset splits to process (train/valid/test)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for audio preprocessing",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional per-split sample cap for debugging",
    )
    parser.add_argument(
        "--feature_dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type used when saving spectrograms",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device for feature extraction. Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cached features for each split",
    )
    return parser.parse_args()


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def prepare_split_dir(split_dir: Path, overwrite: bool):
    if split_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{split_dir} already exists. Rerun with --overwrite to rebuild the cache."
            )
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)


def build_dataset(split: str, config: dict, max_samples: int):
    # Work on a copy so we don't mutate the original training params.
    dataset_training_params = copy.deepcopy(config["training_params"])
    dataset_training_params.setdefault("dataset_params", {})
    if max_samples is not None:
        dataset_training_params["dataset_params"]["max_samples"] = max_samples

    dataset_root = (
        dataset_training_params["training_dataset_path"]
        if split == "train"
        else dataset_training_params.get(
            "evaluation_dataset_path", dataset_training_params["training_dataset_path"]
        )
    )

    args = SimpleNamespace(rank=0)
    return ViMDDataset(dataset_root, dataset_training_params, config["tokenizer_params"], split, args)


def main():
    args = parse_args()
    config = load_config(args.config_file)

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    encoder_params = config["encoder_params"]
    preprocessor = AudioPreprocessing(
        encoder_params["sample_rate"],
        encoder_params["n_fft"],
        encoder_params["win_length_ms"],
        encoder_params["hop_length_ms"],
        encoder_params["n_mels"],
        encoder_params.get("normalize", False),
        encoder_params.get("mean", 0.0),
        encoder_params.get("std", 1.0),
    ).to(device)
    preprocessor.eval()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer = spm.SentencePieceProcessor(config["tokenizer_params"]["tokenizer_path"])

    save_dtype = getattr(torch, args.feature_dtype)

    for split in args.splits:
        dataset = build_dataset(split, config, args.max_samples)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn_pad,
            pin_memory=False,
        )

        split_dir = output_root / split
        prepare_split_dir(split_dir, args.overwrite)

        metadata = {"split": split, "samples": []}
        sample_index = 0

        progress = tqdm(loader, desc=f"Extracting {split}")
        for batch in progress:
            waveforms, targets, waveform_lengths, target_lengths = batch
            waveforms = waveforms.to(device)
            waveform_lengths = waveform_lengths.to(device)

            with torch.no_grad():
                features, feature_lengths = preprocessor(waveforms, waveform_lengths)

            for idx in range(features.size(0)):
                feat_len = int(feature_lengths[idx])
                lbl_len = int(target_lengths[idx])
                spec = features[idx, :, :feat_len].cpu().to(save_dtype)
                label = targets[idx, :lbl_len].cpu().long()
                text = tokenizer.decode(label.tolist())

                file_name = f"{split}_{sample_index:08d}.pt"
                sample_path = split_dir / file_name
                torch.save(
                    {
                        "spectrogram": spec,
                        "label": label,
                        "input_length": feat_len,
                        "label_length": lbl_len,
                        "text": text,
                    },
                    sample_path,
                )

                metadata["samples"].append(
                    {
                        "file": file_name,
                        "frames": feat_len,
                        "label_length": lbl_len,
                        "text": text,
                    }
                )

                sample_index += 1

        metadata_path = split_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

        print(f"Saved {sample_index} samples for split '{split}' to {split_dir}")


if __name__ == "__main__":
    main()
