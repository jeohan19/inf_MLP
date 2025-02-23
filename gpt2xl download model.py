from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Stáhne a uloží model do specifikované složky
model_name = "gpt2-xl"
save_path = "./gpt2_xl_model"

# Stažení tokenizeru a modelu
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Uložení modelu
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
