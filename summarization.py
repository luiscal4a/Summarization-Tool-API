import psutil
import ray

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pickle as pkl
import nltk

import time

checkpoint = "sshleifer/distilbart-cnn-12-6"


tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
summarizer = pipeline("summarization", tokenizer=tokenizer, model=model)
data = {}

nltk.download('punkt')

def getSentences(text):
  return nltk.tokenize.sent_tokenize(text)

def maxSentence(textList):
  return max([len(tokenizer.tokenize(sentence)) for sentence in textList])

def getChunks(textList):
  # initialize
  chunk = ""
  chunks = []
  count = -1
  for sentence in textList:
    count += 1
    combined_length = len(tokenizer.tokenize((chunk+sentence))) # add the no. of sentence tokens to the length counter

    if combined_length  < tokenizer.max_len_single_sentence: # if it doesn't exceed
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
      combined_length = len(tokenizer.tokenize(sentence))
  return chunks

def summarizeAux(inputs, min_len, max_len):
    summary = ""
    
    print(f'Here {len(inputs)}')
    num_cpus = psutil.cpu_count(logical=True) - 4
    ray.init(num_cpus=num_cpus, ignore_reinit_error=True)
    pipe_id = ray.put(summarizer)

    results =  ray.get([summarizeParalel.remote(pipe_id, input, min_len, max_len) for input in inputs])
    ray.shutdown()
    print("results")
    data['summaries'].extend(results)
    summary = ' '.join(results)

    if len(tokenizer.tokenize(summary)) > max_len and len(inputs) > 1:
        print("Dividing")
        summary = divide(summary, min_len, max_len)
    
    return summary

@ray.remote
def summarizeParalel(pipeline, text, min_len, max_len):
    return pipeline(text, min_length=min_len, max_length=max_len)[0]['summary_text'] 

def divide(text, min_len=75, max_len=300):
  sentences = getSentences(text)
  if maxSentence(sentences) > tokenizer.max_len_single_sentence:
    return "Sentence longer than tokenizer"
  else:
    chunks = getChunks(sentences)
    return summarizeAux(chunks, min_len, max_len)
# function
def displayText(content):
    return content

def summarize(content):
  global data
  data = {'summaries': [], 'final_summary': ''}
  final_sum = divide(content)
  data['final_summary'] = final_sum
  return data