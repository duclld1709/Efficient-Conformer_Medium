import glob
from tqdm import tqdm
import os
import torch
import concurrent.futures

audio_max_length, label_max_length = 256000, 256000

split = 'train'
dataset_path = "/mnt/c/Users/hyngu/Data/ASRDataset"

names = glob.glob(dataset_path + "/*/" + split + "/*.wav") 

fw = open("data/" + str(split) + "_wav_names.txt", "a+", encoding="utf8")

def get_name(name):
    if os.path.exists(name + "_len"):
        if torch.load(name + "_len") <= audio_max_length and torch.load(name.replace("wav", "bpe" + "_" + "1024" + "_len")) <= label_max_length:
            fw.write(name + "\n")

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(lambda file: get_name(file), names), total=len(names)))