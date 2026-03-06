import numpy as np
import soundfile as sf
import glob
from tqdm import tqdm
import random
from multiprocessing import Pool

noise_paths_folder = glob.glob("/mnt/c/Users/hyngu/Dataset/dataNoise/*")
random.shuffle(noise_paths_folder)
noise_paths = []

for path in noise_paths_folder:
    files = glob.glob(path + "/*")
    random.shuffle(files)
    try:
        noise_paths += [files[0]]
    except:
        continue

def augmentNoiseAudio(audio_path):

    for noise_path in noise_paths:

        try:

            # Load the main audio file
            audio_data, sample_rate = sf.read(audio_path)

            with open(audio_path.replace(".wav", ".txt"), "r", encoding="utf8") as fr:
                text = fr.readlines()[0].replace("\n", "")

            # Load the noise audio file
            noise_data, _ = sf.read(noise_path)

            # Ensure the noise file matches the length of the main audio file
            if len(noise_data) < len(audio_data):
                # Repeat noise until it matches the length of audio_data
                num_repeats = int(np.ceil(len(audio_data) / len(noise_data)))
                noise_data = np.tile(noise_data, num_repeats)

            # Trim the noise to match the length of audio_data
            noise_data = noise_data[:len(audio_data)]

            # Add the noise to the audio data
            combined_audio = audio_data + noise_data

            # Ensure values are within the proper range (clip to avoid distortion)
            combined_audio = np.clip(combined_audio, -1.0, 1.0)

            noise_wav_name = noise_path.split("/")[-1].replace(".wav", "")

            augment_wav_file = "/".join(audio_path.split("/")[0:-1]) + "/" + audio_path.split("/")[-1].replace(".wav", "") + "_add_noise_" + noise_wav_name + ".wav"
            augment_text_file = "/".join(audio_path.split("/")[0:-1]) + "/" + audio_path.split("/")[-1].replace(".wav", "") + "_add_noise_" + noise_wav_name + ".txt"

            with open(augment_text_file, "w", encoding="utf8") as fw:
                fw.write(text)
            
            sf.write(augment_wav_file, combined_audio, sample_rate)

        except:

            pass

if __name__ == "__main__":
    
    wav_paths = glob.glob("/mnt/c/Users/hyngu/Data/ASRDataset/VLSP2022_Agument2/train/*.wav")
    random.shuffle(wav_paths)
    wav_paths = wav_paths[0:10000]
    with Pool(processes=16) as p:
        list(tqdm(p.imap(augmentNoiseAudio, wav_paths), total=len(wav_paths)))