# Extractive Summarisers
Dissertation code for my 3rd Year Paper on Exploring Text Summarisation

Will perform 2 types of summarisation plus a control case. The first being TextRank (functions correctly) and the other being CNN.

ROUGE is the scoring metric used to evaluate the similarity between the generated 100 word summaries and the 100 word human-written test summaries.

## Preprocessing:
 - Preprocess documents into 'entries', each entry represents a document and its attributes (sentences, parsed_sentences, paired human-written summary)
 - Test and training entries are assigned to test_entries, train_entries in preprocessing.Preprocessing
 - Training scores are assigned to the test data based on ROUGE similarity between each sentence in the train summary and human-written summary.

## Postprocessing:
 - Once generated summaries are obtained, ROUGE similarity is compared with human-written summary and returned.

## Installation:
 - Install requirements
 - Probably need to install rougescore seperately (https://github.com/bdusell/rougescore)
 - Need to download google Word2Vec model (https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit), place in dissertation-text-summarisation/model/
 
## Run:
 - In main function in main class, set train=True in self.evaluate(train=) to train the CNN. S train=False to evaluate and calculate the ROUGE scoring, printing the results.
 - Run main.py for each case
 - 3 Scores are printed after evaluation, CNN, TextRank and control case
 - Can optionally apply stemming by setting the stem parameter in load_dataset(stem=) in the main function (although this is irrlevant to the problem)
  
## Problems:
  - CNN giving near random results. Ideal ROUGE accuracy should be about 47/48%
  
## Paper to cross reference CNN architecture:
  https://ieeexplore.ieee.org/document/7793761
  
  
