# SEO-with-NLP
We propose to implement a search engine that optimizes user search and displays relevant
products as suggested by language models. This will incorporate three modules:
• Spell Check module- implement “Symspell”.
• Classifier or Classification module- implement natural language models (BERT).
• Feature Extraction module- implement SVM to extract features from user strings.

3.1. DATASET

3.1.1. Data Creation
The data preparation task consists of 3 parts that includes data from 3 sources for training
dataset:
1.) Data Scrape from “amazon” or ‘flipkart” to gather titles as strings for category.
2.) Keyword or Search string data, that a user commonly uses to search products on any
commercial websites using Amazon Keyword Tool, available online
(https://keywordtool.io/amazon)
3.) Creating Synthetic Search string - Creating keyword data from titles of each category
manually and tweaking it so that possible combinations of search string can be added in
training dataset
3.2.1 Models
The search engine consists of 2 major modules working in sync with each other:
Spell Check Module: Search string given by user will be auto corrected not in isolation but
keeping in mind context of the query.
Algorithms: Language Model
• Language Model
A probability distribution over words or word sequences is the foundation of a language
model. A language model indicates the likelihood that a particular word sequence is "valid."
Validity in this context has nothing to do with grammatical correctness. The language model
learns that this means that it mimics how people speak (or to be more accurate, write). This is
a crucial point: a language model (like other machine learning models, deep neural networks)
is "just" a tool for condensing a wealth of information in a way that is applicable outside of a
sample.
Classification Module: Product classifier module will category search string and limit search
within specific product category.
Algorithms: Word to Vector, Support vector machine
• Work to Vector
A method for natural language processing called Word2vec was released in 2013. With the
help of a huge text corpus, the word2vec technique employs a neural network model to learn
word associations. Once trained, a model like this can identify terms that are similar or
suggest new words to complete a sentence. As the name suggests, word2vec uses a specific
set of numbers called a vector to represent each unique word. Given that the vectors were
properly selected to capture the semantic and syntactic characteristics of words, the degree of
semantic similarity between the words represented by those vectors may be determined using
a straightforward mathematical function (cosine similarity).

• Spelling Check Module
SymSpell is an algorithm to find all strings within a maximum edit distance from a
huge list of strings in very short time. It can be used for spelling correction and fuzzy
string search. The Symmetric Delete spelling correction algorithm reduces the
complexity of edit candidate generation and dictionary lookup for a given DamerauLevenshtein distance. It is six orders of magnitude faster (than the standard approach
with deletes + transposes + replaces + inserts) and language independent.
For the model to be prepared, the corpus needs to be created out of text titles created
in the data creation step. The text needs to be cleaned using NLTK packages of
Python to be given as input to SymSpell model creation.
3.2.2. Classification Module
The classification model consists of training corpus on BERT or “Bidirectional
Encoder Representations from Transformers” and SVM or “Support Vector
Machines”.
Pre-processing steps:
• Remove all digits from column “Title” or strings.
• Remove all special characters from “Title_processed” in step 1 column.
• Replace double spaces with single spaces and remove any leading and trailing
spaces from strings.
• For removing stopwords and lemmatization we use NLTK library or “Natural
Language Toolkit”. It is a leading platform for building Python programs to work
with human language data.

3.2.4. Feature Extraction Module

This module consists of creating the corpus that has 2 columns:
• Feature tag
• Feature name (example- “Lenovo” is a feature name and “Brand” is its tag)

4.1.5. BERT VS SVM

BERT is a transformer-based model. The pre-trained BERT can be used for two purposes:
• Fine-tuning
• Extracting embedding
You don't need to use an SVM once you're keyed into a BERT architecture. Your BERT
model will generate embeddings and can be fine-tuned (ala ULMfit last layer) to perform
a specific task. You could potentially just use the embeddings and then perform the task
with another model, but the performance would likely not be better. So, how you want to
use BERT remains a choice. But if you can fine-tune the BERT model, it would
generally yield higher performance. But you'll have to validate it based on the
experiments. BERT is undoubtedly a breakthrough in the use of Machine Learning for
Natural Language Processing. The fact that it’s approachable and allows fast fine-tuning
will likely allow a wide range of practical applications in the future.

Output:


![Screenshot 2023-03-21 202759](https://user-images.githubusercontent.com/78313402/226647146-78b97620-a88e-4dc6-908b-bc858d2a1499.png)


