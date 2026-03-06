import torch
import glob
from tqdm import tqdm

torch.set_num_threads(1)

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps, _, read_audio, _, _) = utils

def get_speech_timestamps_vad(audio_path, threshold=0.6, min_silence_duration_ms=100):
    wav = read_audio(audio_path)
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        threshold=threshold, 
        min_silence_duration_ms=min_silence_duration_ms,
        return_seconds=True,  # Return speech timestamps in seconds (default is samples)
    )
    return speech_timestamps

output_folder_path = "/mnt/c/Users/hyngu/Data/ASRDataset/VietBud500/train"

for wav_path in tqdm(glob.glob(output_folder_path + "/*.wav")):
    speech_timestamps = get_speech_timestamps_vad(wav_path)
    if len(speech_timestamps) == 0:
        print(wav_path)