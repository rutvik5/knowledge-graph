import nltk
import sys
import pickle
import os
from collections import defaultdict
import glob
# For Spacy:
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from pprint import pprint
# For custom ER:
import tkinter
import re
# For Coreference resolution
import json
from stanfordcorenlp import StanfordCoreNLP

class StanfordNER:
    def __init__(self):
        self.get_stanford_ner_location()

    def get_stanford_ner_location(self):
        print("Provide (relative/absolute) path to stanford ner package.\n Press carriage return to use './stanford-ner-2018-10-16' as path:") 
        loc = input()
        print("... Running stanford for NER; this may take some time ...")
        if(loc == ''):
            loc = "./stanford-ner-2018-10-16"
        self.stanford_ner_tagger = nltk.tag.StanfordNERTagger(loc+'/classifiers/english.all.3class.distsim.crf.ser.gz',
        loc+'/stanford-ner.jar')

    def ner(self,doc):
        sentences = nltk.sent_tokenize(doc)
        result = []
        for sent in sentences:
            words = nltk.word_tokenize(sent)
            tagged = self.stanford_ner_tagger.tag(words)
            result.append(tagged)
        return result

    def display(self,ner):
        print(ner)
        print("\n")
    
class SpacyNER:
    def ner(self,doc):    
        nlp = en_core_web_sm.load()
        doc = nlp(doc)
        return [(X.text, X.label_) for X in doc.ents]
    
    def ner_to_dict(self,ner):
        """
        Expects ner of the form list of tuples 
        """
        ner_dict = {}
        for tup in ner:
            ner_dict[tup[0]] = tup[1]
        return ner_dict
    
    def display(self,ner):
        print(ner)
        print("\n")

class NltkNER:
    def ner(self,doc):
        pos_tagged = self.assign_pos_tags(doc)
        #chunks = self.split_into_chunks(pos_tagged)
        result = []
        for sent in pos_tagged:
            result.append(nltk.ne_chunk(sent))
        return result

    def assign_pos_tags(self,doc):
        sentences = nltk.sent_tokenize(doc)
        words = [nltk.word_tokenize(sent) for sent in sentences]
        pos_tagged = [nltk.pos_tag(word) for word in words]
        return pos_tagged
    
    def split_into_chunks(self,sentences):
        # This rule says that an NP chunk should be formed whenever the chunker finds an optional determiner (DT) or possessive pronoun (PRP$) followed by any number of adjectives (JJ/JJR/JJS) and then any number of nouns (NN/NNS/NNP/NNPS) {dictator/NN Kim/NNP Jong/NNP Un/NNP}. Using this grammar, we create a chunk parser.
        grammar = "NP: {<DT|PRP\$>?<JJ.*>*<NN.*>+}"
        cp = nltk.RegexpParser(grammar)
        chunks = []
        for sent in sentences:
            chunks.append(cp.parse(sent))
        return chunks

    def display(self,ner):
        print("\n\nTagged: \n\n")
        pprint(ner)
        print("\n\nTree: \n\n ")
        for leaves in ner:
            print(leaves)
            #leaves.draw()
        print("\n")

class CoreferenceResolver:
    def generate_coreferences(self,doc,stanford_core_nlp_path,verbose):
        '''
        pickles results object to coref_res.pickle
        the result has the following structure:
        dict of dict of lists of dicts:  { { [ {} ] } }  -- We are interested in the 'corefs' key { [ { } ] }-- Each list has all coreferences to a given pronoun.
        '''
        nlp = StanfordCoreNLP(stanford_core_nlp_path, quiet =  not verbose)
        props = {'annotators': 'coref', 'pipelineLanguage': 'en'}
        annotated = nlp.annotate(doc, properties=props)
        print("\nannotated\n\n",annotated,"\n\n")
        result = json.loads(annotated)
        # Dump coreferences to a file
        pickle.dump(result,open( "coref_res.pickle", "wb" ))
        # Close server to release memory
        nlp.close()
        return result

    def display_dict(self,result):
        for key in result:
            print(key,":\n",result[key]) 
            print("\n")

    def unpickle(self):
        result = pickle.load(open( "coref_res.pickle", "rb" ))
        return result
    
    def resolve_coreferences(self,corefs,doc,ner,verbose):
        """
        Changes doc's coreferences to match the entity present in ner provided.
        ner must be a dict with entities as keys and names/types as values
        E.g. { "Varun" : "Person" }
        """
        corefs = corefs['corefs']
        if verbose:
            print("Coreferences found: ",len(corefs),"\nThe coreferences are:")
            self.display_dict(corefs)
            print("Named entities:")
            print(ner.keys())

        # replace all corefs in i th coref list with this
        replace_coref_with = []
        
        # Key is sentence number; value is list of tuples. 
        # Each tuple is (reference_dict, coreference number)
        sentence_wise_replacements = defaultdict(list)         # { 0: [ ({},ref#),({},ref#), ...], 1: [({}) ...]... }  

        sentences = nltk.sent_tokenize(doc)
        for index,coreferences in enumerate(corefs.values()):    # corefs : {[{}]} => coreferences : [{}]
            # Find which coreference to replace each coreference with. By default, replace with first reference.
            replace_with = coreferences[0]
            for reference in coreferences:      # reference : {}
                if reference["text"] in ner.keys() or reference["text"][reference["headIndex"]-reference["startIndex"]] in ner.keys():
                    replace_with = reference
                sentence_wise_replacements[reference["sentNum"]-1].append((reference,index))
            replace_coref_with.append(replace_with["text"])  
        
        # sort tuples in list according to start indices for replacement 
        sentence_wise_replacements[0].sort(key=lambda tup: tup[0]["startIndex"]) 

        if verbose:
            for key,val in sentence_wise_replacements.items():
                print("Sent no# ",key)
                for item in val:
                    print(item[0]["text"]," ",item[0]["startIndex"]," ",item[0]["endIndex"]," -> ",replace_coref_with[item[1]]," replacement correl #",item[1], end ="   ||| ")
                print("\n")

        
        #Carry out replacement
        for index,sent in enumerate(sentences):
            # Get the replacements in ith sentence
            replacement_list = sentence_wise_replacements[index]    # replacement_list : [({},int)]
            # Replace from last to not mess up previous replacement's indices
            for item in replacement_list[::-1]:                     # item : ({},int)
                to_replace = item[0]                                # to_replace: {}
                replace_with = replace_coref_with[item[1]]
                replaced_sent = ""
                words = nltk.word_tokenize(sent)
                
                # replace only if what is inted to be replaced is the thing we are trying to replace
                # to_be_replaced = ""
                # for i in range(to_replace["startIndex"],to_replace["endIndex"]):
                #     to_be_replaced  += words[i]
                # if verbose:
                #     print("Intended Replacement: ", to_replace["text"])
                #     print("What's to be replaced: ", to_be_replaced)
                # if to_be_replaced != to_replace["text"]:
                #     if verbose:
                #         print("Texts do not match, skipping replacement")
                #     continue

                if verbose:
                    print("Original: ",sent)
                    print("To replace:", to_replace["text"]," | at:",to_replace["startIndex"],to_replace["endIndex"],end='')
                    print(" With: ",replace_with)
                # Add words from end till the word(s) that need(s) to be replaced
                for i in range(len(words)-1,to_replace["endIndex"]-2,-1):
                    replaced_sent = words[i] + " "+ replaced_sent
                # Replace
                replaced_sent = replace_with + " " + replaced_sent
                # Copy starting sentence
                for i in range(to_replace["startIndex"]-2,-1,-1):
                    replaced_sent = words[i] + " "+ replaced_sent
                if verbose:
                    print("Result: ",replaced_sent,"\n\n")
                sentences[index] = replaced_sent

        result = ""
        for sent in sentences:
            result += sent
        if verbose:
            print("Original text: \n",doc)
            print("Resolved text:\n ",result)
        return result

def resolve_coreferences(doc,stanford_core_nlp_path,ner,verbose):
    coref_obj = CoreferenceResolver()
    corefs = coref_obj.generate_coreferences(doc,stanford_core_nlp_path,verbose)
    #coref.unpickle()
    result = coref_obj.resolve_coreferences(corefs,doc,ner,verbose)
    return result

def main():
    if len(sys.argv) == 1:
        print("Usage:   python3 knowledge_graph.py <nltk/stanford/spacy> [optimized,verbose,nltk,stanford,spacy]")
        return None

    verbose = False
    execute_coref_resol = False
    output_path = "./data/output/"
    ner_pickles_op = output_path + "ner/"
    coref_cache_path = output_path + "caches/"
    coref_resolved_op = output_path + "kg/"
    
    stanford_core_nlp_path = input("\n\nProvide (relative/absolute) path to stanford core nlp package.\n Press carriage return to use './stanford-corenlp-full-2018-10-05' as path:")
    if(stanford_core_nlp_path == ''):
        stanford_core_nlp_path = "./stanford-corenlp-full-2018-10-05"

    file_list = []
    for f in glob.glob('./data/input/*'):
        file_list.append(f)

    for file in file_list:
        with open(file,"r") as f:
            lines = f.read().splitlines()
        
        doc = ""
        for line in lines:
            doc += line

        if verbose:
            print("Read: \n",doc)

        
        for i in range(1,len(sys.argv)):
            if(sys.argv[i] == "nltk"):
                print("\nusing NLTK for NER")
                nltk_ner = NltkNER()
                named_entities = nltk_ner.ner(doc)
                nltk_ner.display(named_entities)
                # ToDo -- Implement ner_to_dict for nltk_ner
                spacy_ner = SpacyNER()
                named_entities = spacy_ner .ner_to_dict(spacy_ner.ner(doc))
            elif(sys.argv[i]=="stanford"):
                print("using Stanford for NER (may take a while):  \n\n\n")
                stanford_ner = StanfordNER()
                tagged = stanford_ner.ner(doc)
                ner = stanford_ner.ner(doc)
                stanford_ner.display(ner)
                # ToDo -- Implement ner_to_dict for stanford_ner
                named_entities = spacy_ner.ner_to_dict(spacy_ner.ner(doc))
            elif(sys.argv[i]=="spacy"):
                print("\nusing Spacy for NER\n")
                spacy_ner = SpacyNER()
                named_entities = spacy_ner.ner(doc)
                spacy_ner.display(named_entities)
                named_entities = spacy_ner.ner_to_dict(named_entities)
            elif(sys.argv[i]=="verbose"):
                verbose = True
            elif(sys.argv[i]=="optimized"):
                execute_coref_resol = True
        
        # Save named entities
        op_pickle_filename = ner_pickles_op + "named_entity_" + file.split('/')[-1].split('.')[0] + ".pickle"
        with open(op_pickle_filename,"wb") as f:
            pickle.dump(named_entities, f)

        if(execute_coref_resol):
            print("\nResolving Coreferences... (This may take a while)\n")
            doc = resolve_coreferences(doc,stanford_core_nlp_path,named_entities,verbose)

        op_filename = coref_resolved_op + file.split('/')[-1]
        with open(op_filename,"w+") as f:
            f.write(doc)
main()