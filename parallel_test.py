# -*- coding: utf-8 -*-

import psutil
import ray

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pickle as pkl
import nltk

import time

import gc

tokenizer = AutoTokenizer.from_pretrained("distilbart")
model = AutoModelForSeq2SeqLM.from_pretrained("distilbart")
summarizer = pipeline("summarization", tokenizer=tokenizer, model=model)

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

def summarizeAux(inputs, isParallel, min_len, max_len):
    summary = ""
    
    if isParallel:
      num_cpus = psutil.cpu_count(logical=True) - 2
      print(f'Number of cpus {num_cpus}')
      ray.init(num_cpus=num_cpus)
      pipe_id = ray.put(summarizer)

      results =  ray.get([summarizeParalel.remote(pipe_id, input, min_len, max_len) for input in inputs])
      ray.shutdown()
      print(f'This was {len(results)} chunks long')
      summary = ' '.join(results)

    else:
      for i in inputs:
        summary += summarizer(i, min_length=min_len, max_length=max_len)[0]['summary_text'] 


    if len(tokenizer.tokenize(summary)) > max_len and len(inputs) > 1:
        print("Dividing")
        summary = divide(summary, isParallel, min_len, max_len)
        
    return summary

@ray.remote
def summarizeParalel(pipeline, text, min_len, max_len):
    auto_garbage_collect()
    return pipeline(text, min_length=min_len, max_length=max_len)[0]['summary_text'] 

def auto_garbage_collect(pct=80.0):
    if psutil.virtual_memory().percent >= pct:
        gc.collect()

def divide(text, isParallel, min_len=75, max_len=300):
  sentences = getSentences(text)
  if maxSentence(sentences) > tokenizer.max_len_single_sentence:
    return "Sentence longer than tokenizer"
  else:
    chunks = getChunks(sentences)
    return summarizeAux(chunks, isParallel, min_len, max_len)

def generate_string(number):
  return "Catching up with a new or well known topic hasn’t been both easier and harder before. It doesn’t matter if you are approaching a new knowledge field for the first time or you are already a regular, you encounter very similar problems: lots of publications to read, less helpful filtering tools and most likely the lack of time to cover everything.It’s easy to find online several articles and forums discussions about last years’ exponential growth of papers published each year, mostly arguing about how this trend is leading to an excess of publications and why this is even counterproductive to the scientific community. The monthly submissions stats from arXiv back these points, with an all time high in March 2021 with 16k new submissions, extrapolating this to a full year we are talking about 192k yearly new submissions, or 526 daily publications in average.The best hope of anyone interested in being up to date in this rush of new scientific content is to rely on others to do the filtering / reviewing job and condense the most information possible in a more convenient & consumable format, we are talking about a summary.The task of reading a document and producing a summary is a common task in learning to demonstrate both reading comprehension and writing ability. When we as humans summarize a piece of text, we usually read it entirely to develop our understanding, and then write a summary highlighting its main points. Since computers lack human knowledge and language capability, it makes automatic text summarization a very difficult and non-trivial task. Most of the NLP approaches model this problem as a classification problem which outputs whether to include a sentence in the summary or not. Other approaches have used topic information, Latent Semantic Analysis (LSA), Sequence to Sequence models, Reinforcement Learning and Adversarial processes. But in general we can distinguish between two different ways of approaching the problem: extraction and abstraction. Extractive summarization is based on the idea of picking sentences directly from the original document based on a scoring function to form a coherent summary. This method identifies important sections of the text, crops them out and stitches together portions of the content to produce a condensed version. Abstractive summarization aims to produce the summary text by interpreting the original source with advanced NLP techniques in order to generate a new shorter text, with parts that may not appear on the original document, that conveys the most critical information of the text. Sometimes this requires rephrasing sentences and incorporating information from full text to generate summaries in a similar way to human-written summaries.This abstractive text summarization is one of  the most challenging tasks in natural processing, as it involves the understanding of long pieces, information compression, and language generation. Since it can be considered a sequence mapping task where the source text must be mapped to the target summary, latest methods take advantage of recent developments in deep learning, specifically on sequence to sequence models.Catching up with a new or well known topic hasn’t been both easier and harder before. It doesn’t matter if you are approaching a new knowledge field for the first time or you are already a regular, you encounter very similar problems: lots of publications to read, less helpful filtering tools and most likely the lack of time to cover everything.Catching up with a new or well known topic hasn’t been both easier and harder before.Catching up with a new or well known topic hasn’t been both easier and harder before.Catching up with a new or well known topic hasn’t been both easier and harder before.Catching up with a new or well known topic hasn’t been both easier and harder before.Catching up with a new or well known topic hasn’t been both easier and harder before.Catching up with a new or well known topic hasn’t been both easier and harder before.Catching up with a new or well known topic hasn’t been both easier and harder before.Catching up with a new or well known topic hasn’t been both easier and harder before.Catching up with a new or well known topic hasn’t been both easier and harder before.Catching up with a new or well known topic hasn’t been both easier and harder before.Catching up with a new or well known topic hasn’t been both easier and harder before.Catching up with a new or well known topic hasn’t been both easier and harder before.Catching up with a new or well known topic hasn’t been both easier and harder before.Catching up with a new or well known topic hasn’t been both easier and harder before.Catching up with a new or well known topic hasn’t been both easier and harder before.Catching up with a new or well known topic hasn’t been both easier and harder before.Catching up with a new or well known topic hasn’t been both easier and harder before. " * number

tests = [1,2,3,4,5,10,14,15,20]
tests = [50]

for test in tests:
  start = time.time()
  hello = generate_string(test)
  print(f'--TEST FOR {test} DIVISIONS')
  print(len(tokenizer.tokenize(hello)))
  divide(hello, True)
  end = time.time()
  print(f'Time {end - start}')