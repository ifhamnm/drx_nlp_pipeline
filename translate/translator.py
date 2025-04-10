from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from utils.logger import log_tokens_per_second

model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

def translate_text(text, src_lang="auto", target_lang="en"):
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt", truncation=True)

    with log_tokens_per_second("🌍 Translating"):
        generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(target_lang))

    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
