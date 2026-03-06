from multiprocessing import Pool
import os
from tqdm import tqdm
from functions import *
import json
import pickle
import torchaudio
import concurrent.futures
import glob

# Load Config
with open("configs/EfficientConformerCTCSmall.json") as json_config:
    config = json.load(json_config)

device = torch.device("cpu")

tokenizer_params = {
    "vocab_type": "bpe",
    "vocab_size": 1024
}

model = create_model(config).to(device)

label_paths = []
sentences = []

txt_files = glob.glob("/mnt/c/Users/hyngu/Data/ASRDataset/VLSP2022_Agument2/" + "val/*.txt") + glob.glob("/mnt/c/Users/hyngu/Data/ASRDataset/VLSP2022_Agument2/" + "train/*.txt")

def process_text_file(file_path):
    """Đọc file và chuẩn hóa văn bản."""
    local_label_paths = []
    local_sentences = []
    if os.path.exists(file_path.replace(".txt", ".wav")):
        try:
            with open(file_path, "r", encoding="utf8") as f:
                for line in f.readlines():
                    text = " ".join(line.split())
                    if len(text.split()) > 1:
                        label_path = file_path.replace(file_path.split("/")[-1], "") + \
                                    file_path.split("/")[-1].replace(".txt", "") + \
                                    "." + tokenizer_params["vocab_type"] + "_" + str(tokenizer_params["vocab_size"])
                        local_label_paths.append(label_path)
                        local_sentences.append(text.lower())
        except Exception as e:
            print(f"\nError reading {file_path}: {e}")
    
    return local_sentences, local_label_paths

def process_sample(item):
    """Hàm xử lý từng mẫu dữ liệu"""
    try:
        sentence, label_path = item[0], item[1]

        # Tokenize và lưu label
        label = torch.LongTensor(model.tokenizer.encode(sentence))

        # Lưu độ dài nhãn
        label_length = label.size(0)
        
        torch.save(label, label_path)
        torch.save(label_length, label_path + "_len")

        # Lưu độ dài audio
        audio_length = torchaudio.load(label_path.split("." + "bpe" + "_" + str("1024"))[0] + ".wav")[0].size(1)
        torch.save(audio_length, label_path.split("." + "bpe" + "_" + str("1024"))[0] + ".wav_len")
        
    except Exception as e:
        print(f"\nError processing {label_path}: {e}")

if __name__ == "__main__":

    # Đọc file song song bằng ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(lambda file: process_text_file(file), txt_files), total=len(txt_files)))

    # Gộp kết quả từ các thread
    for local_sentences, local_label_paths in results:
        sentences.extend(local_sentences)
        label_paths.extend(local_label_paths)

    # Giả sử bạn đã có 2 mảng sentences và label_paths
    with open('dataVLSP2022Agument.pkl', 'wb') as f:
        pickle.dump((sentences, label_paths), f)

    with open('dataVLSP2022Agument.pkl', 'rb') as f:
        sentences, label_paths = pickle.load(f)
    
    items = []
    for i, sentence in enumerate(sentences):
        items.append([sentence, label_paths[i]])
    with Pool(processes=8) as pool:
        list(tqdm(pool.imap(process_sample, items), total=len(items)))