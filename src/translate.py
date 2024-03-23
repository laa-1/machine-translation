from transformers import BertTokenizer
from parameter import *


def translate(text):
    model = torch.load(model_path.format(100000)).to(device)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    src = tokenizer.encode(text, truncation=True, max_length=tokens_max_length - 2, add_special_tokens=False)
    src = torch.tensor([[bos_token_id] + src + [eos_token_id] + [pad_token_id for _ in range(tokens_max_length - 2 - len(src))]], device=device)
    tgt = torch.tensor([[bos_token_id]], device=device)
    model.eval()
    for i in range(tokens_max_length):
        out = model(src, tgt)
        token_id = torch.argmax(out, dim=-1)
        tgt = torch.concat([tgt, token_id.unsqueeze(0)], dim=1)
        if token_id == eos_token_id:
            break
    result = tokenizer.decode(tgt.squeeze(), skip_special_tokens=True).replace(" ", "")
    print(result)


if __name__ == '__main__':
    print("")
    translate("Hello, world!")
    translate("I'm come from China.")
    translate("I am a translation model in English translation.")
    translate("Nice to meet you.")
