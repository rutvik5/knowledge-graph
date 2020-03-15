# From unstructured text to knowledge graph

The project is a complete end-to-end solution for generating knowledge graphs from unstructured data. NER can be run on input by either NLTK, Spacy or Stanford APIs. Optionally, coreference resolution can be performed which is done by python wrapper to stanford's core NLP API. Relation extraction is then done using stanford's open ie. Lastly, post-processing is done to get csv file which can be uploaded to graph commons to visualize the knowledge graph.

More details can be found in the Approach folder.

## Running the code

1. Clone Repository
2. Ensure your system is setup properly (Refer Setup instructions below)
3. Put your input data files (.txt) in data/input
4. Run knowledge_graph.py       
    `python3 knowledge_graph.py spacy`
    You can provide several arguments to knowledge_graph.py. For a more detailed list, refer the running knowledge_graph.py section below
5. Run relation_extractor.py
    `python3 relation_extractor.py`
6. Run create_structured_csv
    `python3 create_structured_csv.py`
7. The resultant csv is available in data/results folder

## Setup

The following installation steps are written w.r.t. linux operating system and python3 language.

1. Create a new python3 virtual environment:  
    `python3 -m venv <path_to_env/env_name>`
2. Switch to the environment:  
    `source path_to_env/env_name/bin/activate`
3. Install Spacy:  
    `pip3 install spacy`
4. Install en_core_web_sm model for spacy:  
    `python3 -m spacy download en_core_web_sm`
5. Install nltk:  
    `pip3 install nltk`
6. Install required nltk data. Either install required packages individually or install all packages by using  
    `python -m nltk.downloader all`  
    Refer: https://www.nltk.org/data.html
7. Install stanfordcorenlp python package:  
    `pip3 install stanfordcorenlp`
8. Download and unzip stanford-corenlp-full:   
   https://stanfordnlp.github.io/CoreNLP/download.html
9.  Download and setup stanford ner: https://nlp.stanford.edu/software/CRF-NER.shtml#Download as described in NLTK documentation: http://www.nltk.org/api/nltk.tag.html#module-nltk.tag.stanford  (Not required if already present due to git clone)
10. Download and unzip stanford open-ie (Not required if already present due to git clone)
11. Install python-tk:  
    `sudo apt-get install python3-tk`
12. Install pandas:  
    `pip3 install pandas`
### knowledge_graph.py

Performs Named Entity Recognition (NER) on input data by using either NLTK, Spacy or Stanford (or all of them). Also performs coreference resolution. The coreference is used by relation_extractor.py . The recognised NER are used by create_structured_csv.py

##### Running knowledge_graph.py

Will only run on linux like operating systems, with paths like abc/def/file.txt

Please note that coreference resolution server requires around 4GB of free system RAM to run. If this is not available, stanford server may stop with an error or thrashing may cause program to run very slowly.

`python3 knowledge_graph.py <options>` 

options:
 
- nltk           runs Named Entity Recognition using custom code written with help of NLTK
- stanford       runs NER using stanford's library
- spacy          uses spacy's pre-trained models for NER
- verbose        to get detailed output
- optimized      run coreference resolution to get better output. This will increase time taken significantly. 
                 Also will impose a limit on size of each file; so data may need to be split amongst files.

e.g.:

`python3 knowledge_graph.py optimized verbose nltk spacy`  
will o/p ner via nltk and spacy, and perform coreference resolution


##### inputs to knowledge_graph.py

The input unstructured data files must be in ./data/input folder. I.e. data folder must be in same dir as knowledge_graph.py

##### outputs from knowledge_graph.py

data/output/ner     ---  contains recognised named entities  
data/output/caches  --- Intended to contain result pickles of coreferences obtained   by stanford's core nlp  
data/output/kg      --- contains input files with coreferences resolved 