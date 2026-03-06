import glob
from multiprocessing import Pool
from tqdm import tqdm
from shutil import move
import os
import random

agument_folder_path = "/mnt/c/Users/hyngu/Data/ASRDataset/VLSP2022_Agument2/train"
os.makedirs(agument_folder_path, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

def convert_bit_rate(audio_path):
    os.system('ffmpeg -loglevel quiet -i ' + audio_path + ' -b:a 128k ' + audio_path.replace(".wav", "_128k.wav"))

def speed_up(audio_path):
    try:
        with open(audio_path.replace(".wav", ".txt"), "r", encoding="utf8") as fr:
            text = fr.readlines()[0].replace("\n", "")
        os.system('ffmpeg -loglevel quiet -i ' + audio_path + ' -filter:a "atempo=1.1" ' + agument_folder_path + "/" + audio_path.split("/")[-1].replace(".wav", "_sp11.wav") + " -y")
        with open(agument_folder_path + "/" + audio_path.split("/")[-1].replace(".wav", "_sp11.txt"), "w", encoding="utf8") as fw:
            fw.write(text)
    except:
        pass

def remove_(audio_path):
    if not audio_path.__contains__("_16k.wav"):
        os.remove(audio_path)

def rename_(audio_path):
    move(audio_path, audio_path.replace("_16k.wav", ".wav"))

if __name__ == "__main__":
    wav_paths = glob.glob("/mnt/c/Users/hyngu/Data/ASRDataset/VLSP2022/train/*")
    random.shuffle(wav_paths)
    wav_paths = wav_paths[0:10000]
    with Pool(processes=16) as p:
        list(tqdm(p.imap(speed_up, wav_paths), total=len(wav_paths)))