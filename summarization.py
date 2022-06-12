from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pickle as pkl
import nltk
import json
import requests
from os import path

from config import API_TOKEN

headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/"

checkpoint_distilbart = "sshleifer/distilbart-cnn-12-6"
checkpoint_prophetnet = "microsoft/prophetnet-large-uncased-cnndm"

model_key = {"distilbart": checkpoint_distilbart, "prophetnet":checkpoint_prophetnet}
response_key = {"distilbart": "summary_text", "prophetnet":"generated_text"}

model = {"distilbart": None, "prophetnet":None}
tokenizer = {"distilbart": None, "prophetnet":None}
summarizer = None

data = {}

def getSentences(text, model_name):
  return nltk.tokenize.sent_tokenize(text)

def maxSentence(textList, model_name):
  return max([len(tokenizer[model_name].tokenize(sentence)) for sentence in textList])

def getChunks(textList, model_name):
  # initialize
  chunk = ""
  chunks = []
  count = -1
  for sentence in textList:
    count += 1
    combined_length = len(tokenizer[model_name].tokenize((chunk+sentence))) # add the no. of sentence tokens to the length counter

    if combined_length  < tokenizer[model_name].max_len_single_sentence: # if it doesn't exceed
      chunk += sentence + " " # add the sentence to the chunk

      # if it is the last sentence
      if count == len(textList) - 1:
        chunks.append(chunk.strip()) # save the chunk
      
    else: 
      chunks.append(chunk.strip()) # save the chunk
      # reset 
      chunk = ""

      # take care of the overflow sentence
      chunk += sentence + " "
      combined_length = len(tokenizer[model_name].tokenize(sentence))
  return chunks

def query(payload, model_name):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL+model_key[model_name], headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

def summarizeAux(inputs, isAPI, model_name, min_len, max_len):
    summary = ""
    iteration = []
    
    if isAPI:
      for i, input in enumerate(inputs):
        sum_query = { "inputs" : input, "parameters" : {"min_length": min_len, "max_length": max_len}}
        print(query(sum_query, model_name))
        summary_aux = query(sum_query, model_name)[0][response_key[model_name]].replace('[X_SEP] ', '')
        iteration.append(summary_aux)
        print(f'{i+1}/{len(inputs)} {summary_aux}')
        summary = summary + summary_aux

    else:
      for i, input in enumerate(inputs):
        summary_aux = summarizer(input, min_length=min_len, max_length=max_len)[0]["summary_text"].replace('[X_SEP] ', '')
        iteration.append(summary_aux)
        print(f'{i+1}/{len(inputs)} {summary_aux}')
        summary = summary + summary_aux
    
    data['summaries'].append(iteration)

    if len(tokenizer[model_name].tokenize(summary)) > max_len and len(inputs) > 1:
        print("Dividing")
        summary = divide(summary, isAPI, model_name, min_len, max_len)
    
    return summary


def divide(text, isAPI, model_name, min_len=75, max_len=300):
  sentences = getSentences(text, model_name)
  if maxSentence(sentences, model_name) > tokenizer[model_name].max_len_single_sentence:
    return "Sentence longer than tokenizer"
  else:
    chunks = getChunks(sentences, model_name)
    return summarizeAux(chunks, isAPI, model_name, min_len, max_len)

def initialize_model(model_name):
  global model
  model[model_name] = AutoModelForSeq2SeqLM.from_pretrained(model_key[model_name])

def initialize_tokenizer(model_name):
  global tokenizer
  tokenizer[model_name] = AutoTokenizer.from_pretrained(model_key[model_name])


def summarize(content, model_name, min_length, max_length):
  global data, model, tokenizer, summarizer
  data = {'summaries': [], 'final_summary': ''}
  isAPI = API_TOKEN != ""

  if tokenizer[model_name] == None:
    if path.exists(f'./models/{model_name}'):
      tokenizer[model_name] = AutoTokenizer.from_pretrained(f'./models/{model_name}')
      if not isAPI:
        model[model_name] = AutoModelForSeq2SeqLM.from_pretrained(f'./models/{model_name}')
    else:
      initialize_tokenizer(model_name)
      if not isAPI:
        initialize_model(model_name)
    if not isAPI:
      summarizer = pipeline("summarization", tokenizer=tokenizer[model_name], model=model[model_name])

  final_sum = divide(content, isAPI, model_name, min_length, max_length)
  data['final_summary'] = final_sum

  return data
