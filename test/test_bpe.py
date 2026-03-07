import sentencepiece as spm

# 1. Khởi tạo và load model
sp = spm.SentencePieceProcessor()
sp.load(r'D:\Duc_Data\Study\FPT_University_Course\SPRING26_Semester_7\DAT301\code\review_research\Efficient-Conformer_Medium\datasets\Vietnamese\vi_bpe_1024.model')

# 2. Lấy tổng số lượng vocab (kích thước bộ từ điển)
vocab_size = sp.get_piece_size()

# 3. Lấy token tại index cuối cùng (index tính từ 0 nên là vocab_size - 1)
last_token = sp.id_to_piece(vocab_size - 1)

print(f"Tổng số vocab: {vocab_size}")
print(f"Ký tự (token) cuối cùng trong vocab là: '{last_token}'")

print("--- 10 token đầu tiên trong vocab ---")
for i in range(10):
    print(f"Index {i}: '{sp.id_to_piece(i)}'")

print(f"--- 10 token cuối cùng trong bộ từ điển (Tổng size: {vocab_size}) ---")

# Duyệt từ index (vocab_size - 10) đến (vocab_size - 1)
for i in range(0, vocab_size):
    token = sp.id_to_piece(i)
    print(f"Index {i}: '{token}'")

# TEST TOKENIZER
# =============================

text = "hôm nay tôi đang học mô hình nhận dạng tiếng nói"

print("\n===== TEST TOKENIZER =====")
print("Input text:", text)

# Encode -> token pieces
pieces = sp.encode(text, out_type=str)
print("\nToken pieces:")
print(pieces)

# Encode -> token ids
ids = sp.encode(text, out_type=int)
print("\nToken IDs:")
print(ids)

# Decode lại
decoded = sp.decode(ids)
print("\nDecoded text:")
print(decoded)