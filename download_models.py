from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

checkpoint_distilbart = "sshleifer/distilbart-cnn-12-6"
checkpoint_prophetnet = "microsoft/prophetnet-large-uncased-cnndm"

def save_model(checkpoint):
  tokenizer = AutoTokenizer.from_pretrained(checkpoint)
  model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

  tokenizer.save_pretrained(f"./models/{checkpoint.split('/')[1].split('-')[0]}")
  model.save_pretrained(f"./models/{checkpoint.split('/')[1].split('-')[0]}")

  del(tokenizer)
  del(model)

save_model(checkpoint_distilbart)
save_model(checkpoint_prophetnet)
