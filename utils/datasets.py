# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch
import torchaudio

# Other
import glob
import io
import json
from pathlib import Path
from typing import Dict, List

import sentencepiece as spm
import soundfile as sf
from tqdm import tqdm

import os
os.environ["HF_HOME"] = r"D:\hf_datasets"
os.environ["HF_DATASETS_CACHE"] = r"D:\hf_datasets\datasets"
os.environ["TRANSFORMERS_CACHE"] = r"D:\hf_datasets\models"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\hf_datasets\hub"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:  # pragma: no cover - optional dependency
    from datasets import Audio, load_dataset
except ImportError:  # pragma: no cover - handled in ViMDDataset
    Audio = None
    load_dataset = None


class VietnameseDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, training_params, tokenizer_params, split, args):

        self.names = glob.glob(dataset_path + "/*/" + split + "/*.wav")
        self.vocab_type = tokenizer_params["vocab_type"]
        self.vocab_size = str(tokenizer_params["vocab_size"])
        self.lm_mode = training_params.get("lm_mode", False)
        self.split = split

        if split == "train":
            self.names = self.filter_lengths(
                training_params["train_audio_max_length"],
                training_params["train_label_max_length"],
                args.rank,
            )
        else:
            self.names = self.filter_lengths(
                training_params["eval_audio_max_length"],
                training_params["eval_audio_max_length"],
                args.rank,
            )

    def __getitem__(self, i):

        if self.lm_mode:
            return [
                torch.load(self.names[i].split(".wav")[0] + "." + self.vocab_type + "_" + self.vocab_size)
            ]
        else:
            return [
                torchaudio.load(self.names[i])[0],
                torch.load(self.names[i].split(".wav")[0] + "." + self.vocab_type + "_" + self.vocab_size),
            ]

    def __len__(self):

        return len(self.names)

    def filter_lengths(self, audio_max_length, label_max_length, rank=0):

        if audio_max_length is None or label_max_length is None:
            return self.names

        if rank == 0:
            print("LibriSpeech dataset filtering")
            print(
                "Audio maximum length : {} / Label sequence maximum length : {}".format(
                    audio_max_length, label_max_length
                )
            )
            self.names = tqdm(self.names)

        if self.split == "train":
            with open("data/train_wav_names.txt", "r", encoding="utf8") as fr:
                names1 = fr.readlines()
            with open("data/train_agument_wav_names.txt", "r", encoding="utf8") as fr:
                names2 = fr.readlines()
            names = names1 + names2
            print("Total " + str(self.split) + " examples: " + str(len(names)))
            return [name.replace("\n", "") for name in names]
        else:
            with open("data/val_wav_names.txt", "r", encoding="utf8") as fr:
                names = fr.readlines()
            print("Total " + str(self.split) + " examples: " + str(len(names)))
            return [name.replace("\n", "") for name in names]


class ViMDDataset(torch.utils.data.Dataset):
    """Loads the ViMD dataset stored either locally in parquet shards or from the Hugging Face Hub."""

    def __init__(self, dataset_path, training_params, tokenizer_params, split, args):
        if load_dataset is None or Audio is None:
            raise ImportError(
                "The 'datasets' package is required for ViMDDataset. Install it via `pip install datasets`."
            )

        dataset_cfg: Dict[str, str] = training_params.get("dataset_params", {})
        local_dir = dataset_cfg.get("local_data_dir") or dataset_path
        self.local_data_dir = Path(local_dir).expanduser() if local_dir else None
        if self.local_data_dir and not self.local_data_dir.exists():
            self.local_data_dir = None

        self.split = split
        self.text_column = dataset_cfg.get("text_column", "text")
        self.audio_column = dataset_cfg.get("audio_column", "audio")
        self.lowercase = dataset_cfg.get("lowercase", True)
        self.target_sr = dataset_cfg.get("target_sample_rate", training_params.get("sample_rate", 16000))
        self.return_transcript = dataset_cfg.get("include_transcript", False)
        self.max_samples = dataset_cfg.get("max_samples", None)

        tokenizer_path = tokenizer_params.get("tokenizer_path")
        if tokenizer_path is None or not Path(tokenizer_path).exists():
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}.")
        self.tokenizer = spm.SentencePieceProcessor(tokenizer_path)

        if self.local_data_dir:
            pattern = dataset_cfg.get("split_pattern", "{split}-*.parquet").format(split=split)
            shard_paths = sorted(self.local_data_dir.glob(pattern))
            if not shard_paths:
                raise FileNotFoundError(
                    f"No parquet files matching '{pattern}' were found under {self.local_data_dir}."
                )
            data_files = [str(p) for p in shard_paths]
            self.dataset = load_dataset("parquet", data_files=data_files, split="train")
        else:
            hf_name = dataset_cfg.get("hf_dataset_name")
            if hf_name is None:
                if dataset_path and not Path(dataset_path).exists():
                    hf_name = dataset_path
                else:
                    hf_name = "nguyendv02/ViMD_Dataset"
            load_kwargs = {}
            if dataset_cfg.get("hf_config"):
                load_kwargs["name"] = dataset_cfg["hf_config"]
            if dataset_cfg.get("hf_data_dir"):
                load_kwargs["data_dir"] = dataset_cfg["hf_data_dir"]
            if dataset_cfg.get("hf_cache_dir"):
                load_kwargs["cache_dir"] = dataset_cfg["hf_cache_dir"]
            self.dataset = load_dataset(hf_name, split=split, **load_kwargs)

        keep_columns = [col for col in [self.audio_column, self.text_column] if col in self.dataset.column_names]
        self.dataset = self.dataset.select_columns(keep_columns)
        self.dataset = self.dataset.cast_column(self.audio_column, Audio(decode=False))

        if self.max_samples is not None:
            self.dataset = self.dataset.select(range(min(self.max_samples, len(self.dataset))))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        audio_data = item[self.audio_column]
        transcript = (item.get(self.text_column, "") or "").strip()

        waveform = self._load_audio(audio_data)
        label_text = transcript.lower() if self.lowercase else transcript
        label_ids = self.tokenizer.encode(label_text)
        label = torch.LongTensor(label_ids)

        if self.return_transcript:
            return [waveform, label, transcript]
        return [waveform, label]

    def _load_audio(self, audio_dict):
        if "array" in audio_dict and audio_dict["array"] is not None:
            array = audio_dict["array"]
            sr = audio_dict.get("sampling_rate", self.target_sr)
        elif "bytes" in audio_dict:
            with io.BytesIO(audio_dict["bytes"]) as buffer:
                array, sr = sf.read(buffer, dtype="float32")
        else:
            raise KeyError("Audio column must provide either 'array' or 'bytes'.")

        waveform = torch.from_numpy(array)
        if waveform.dim() == 2:
            waveform = waveform.transpose(0, 1).contiguous()
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.dim() > 2:
            waveform = waveform.reshape(waveform.size(0), -1)

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)

        return waveform


class CachedFeatureDataset(torch.utils.data.Dataset):
    """Dataset wrapper that reads pre-extracted spectrogram features stored on disk."""

    _DTYPE_MAP = {
        "float32": torch.float32,
        "float16": torch.float16,
        "float64": torch.float64,
    }

    def __init__(self, cache_root, training_params, tokenizer_params, split, args):
        cache_root = Path(cache_root) if cache_root else None
        if cache_root is None or not cache_root.exists():
            raise FileNotFoundError(f"Cached feature directory not found: {cache_root}")

        dataset_cfg = training_params.get("cached_feature_params", {})
        dtype = dataset_cfg.get("dtype", "float32").lower()
        if dtype not in self._DTYPE_MAP:
            raise ValueError(
                f"Unsupported dtype '{dtype}'. Choose from {list(self._DTYPE_MAP.keys())}."
            )
        self.torch_dtype = self._DTYPE_MAP[dtype]

        self.split = split
        self.split_dir = cache_root / split if split else cache_root
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Cached feature split directory not found: {self.split_dir}")

        metadata_path = self.split_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.samples: List[Dict[str, int]] = metadata.get("samples", metadata)
        else:
            self.samples = [{"file": file.name} for file in sorted(self.split_dir.glob("*.pt"))]

        if not self.samples:
            raise RuntimeError(f"No cached feature files found in {self.split_dir}")

        self.frame_lengths = [sample.get("frames") or sample.get("input_length") for sample in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_meta = self.samples[index]
        file_name = sample_meta.get("file")
        if not file_name:
            raise KeyError("Each metadata entry must include a 'file' field pointing to the .pt sample.")
        sample_path = self.split_dir / file_name
        if not sample_path.exists():
            raise FileNotFoundError(sample_path)

        data = torch.load(sample_path, map_location="cpu")
        spectrogram = data["spectrogram"].to(self.torch_dtype)
        label = data["label"].long()
        input_length = torch.tensor(data["input_length"], dtype=torch.long)
        label_length = torch.tensor(data["label_length"], dtype=torch.long)

        return [spectrogram, label, input_length, label_length]
