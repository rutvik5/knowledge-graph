# coding: utf-8

import nltk
from nltk.tag.stanford import StanfordNERTagger

# Optional
# import os
# java_path = "/usr/lib/jvm/java-8-oracle"
# os.environ['JAVA_HOME'] = java_path

sentence = u"Rajat, Rutvik, and Varun are in Raleigh"

jar = '/Users/rajat/Desktop/Knowledge_Graph/stanford-ner-2018-10-16/stanford-ner.jar'
model = '/Users/rajat/Desktop/Knowledge_Graph/stanford-ner-2018-10-16/dummy-ner-model.ser.gz'

ner_tagger = StanfordNERTagger(model, jar, encoding='utf8')

words = nltk.word_tokenize(sentence)
print(ner_tagger.tag(words))