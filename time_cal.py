import sox
import glob
from multiprocessing import Pool
from tqdm import tqdm

def hasNumber(text):
    text = text.replace(",", " ")
    text = text.replace(".", " ")
    text = text.replace("?", " ")
    text = text.replace("%", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("<", " ")
    text = text.replace(">", " ")
    text = text.replace("/", " ")
    text = " ".join(text.split())
    return any(char.isdigit() for char in text)

def _duration_file_path(path):
    duration = sox.file_info.duration(path)
    if duration is not None:
        return sox.file_info.duration(path)
    return 0.0

if __name__ == "__main__":
    wav_paths = glob.glob("/mnt/c/Users/hyngu/Data/ASRDataset/*/*/*.wav")
    print(len(wav_paths))
    with Pool(processes=16) as p:
        list_durations = list(tqdm(p.imap(_duration_file_path, wav_paths), total=len(wav_paths)))
    total_durations = 0
    for duration in list_durations:
        total_durations += duration
    print("Total hours: " + str(total_durations / 3600))
