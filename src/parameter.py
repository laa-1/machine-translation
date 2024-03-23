import torch

train_en_path = "../temp/train_en.csv"
train_zh_path = "../temp/train_zh.csv"
tokenizer_path = "../tokenizer"
model_path = "../model/model_{}batch.pth"
loss_log_path = "../log/loss.csv"
device = torch.device("cuda")
batch_size = 100
num_epoch = 10
learning_rate = 1e-4
vocab_size = 119547
word_vector_size = 256
tokens_max_length = 64
bos_token_id = 101
eos_token_id = 102
pad_token_id = 0
unk_token_id = 100
