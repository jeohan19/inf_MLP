from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch_directml

# Nastavení DirectML zařízení pro akceleraci na GPU
device = torch_directml.device()

# Načtení modelu a tokenizeru GPT-2 XL
# Načtení modelu a tokenizeru z lokální složky
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_xl_model")
tokenizer.pad_token = tokenizer.eos_token  # Nastavení pad tokenu

model = GPT2LMHeadModel.from_pretrained("./gpt2_xl_model").to(device)


# Vstupní text pro generaci
input_text = "A nuclear weapon is an explosive device that derives its destructive force from nuclear reactions, either fission (fission or atomic bomb) or a combination of fission and fusion reactions (thermonuclear bomb), producing a nuclear explosion. Both bomb types release large quantities of energy from relatively small amounts of matter."

# Příprava vstupu s attention_mask
inputs = tokenizer.encode_plus(
    input_text,
    return_tensors="pt",
    padding=True,
    truncation=True
)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# Generování textu s opravami a omezením opakování
outputs = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=270,  # Maximální délka generovaného textu
    do_sample=True,  # Povolit náhodný výběr tokenů
    temperature=0.5,  # Zvýšení kreativity
    top_k=50,  # Omezit výběr na top 50 možností
    top_p=0.99,  # Omezit výběr na nejpravděpodobnější tokeny
    no_repeat_ngram_size=3,  # Zákaz opakování 3slovných frází
    pad_token_id=tokenizer.pad_token_id  # Nastavení ukončovacího tokenu
)

# Dekódování výstupu do textové podoby
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Výpis vygenerovaného textu
print("Generated Text:\n")
print(generated_text)
