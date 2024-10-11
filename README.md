# **Information-Retrieval based Question and Answer System**

## Overview:
This project focuses on developing a question-and-answer (QA) system using Natural Language Processing (NLP) techniques to extract information from news articles. The system works by identifying relevant text phrases directly tied to a user's inquiry, with a focus on fact-based, single-sentence questions, such as "Who is the vice chairman of Samsung?". The system follows an Information Retrieval (IR) approach, using question processing, passage retrieval, and answer extraction. It incorporates NLP techniques like Named Entity Recognition (NER), dependency parsing, coreference resolution, and TF-IDF.

1. **Question Processing**: The system begins by analyzing the user’s question to identify its key elements. Techniques such as tokenization, part-of-speech tagging, and Named Entity Recognition (NER) are applied to understand the nature of the question. For example, if the question asks for a date, location, or person's name, the system detects these specifics, allowing it to focus on relevant sections of text.

2. **Passage Retrieval**: Once the system understands the question, it searches through a collection of news articles using Information Retrieval (IR) techniques. The IR component leverages methods like TF-IDF (Term Frequency-Inverse Document Frequency) to find the most relevant passages or articles. This ensures the system narrows down the content to the most pertinent sections related to the query.

3. **Answer Extraction**: After retrieving the relevant passage, the system extracts the answer. This is done using various NLP techniques such as syntactic parsing and dependency parsing to analyze the structure of the sentence. The system is primarily designed to extract noun phrases, which are often direct answers to fact-based questions. For example, if the question is "Who is the vice chairman of Samsung?", the system will extract the noun phrase that corresponds to the answer. Additionally, a comparative analysis with a BERT model is utilised to measure the effectiveness of the method.

## Key Methods Utilised:
### 1. **Question Processing**
This phase involves several NLP techniques to break down and understand the user’s query:
- **Focus Detection**: Uses **syntactic parsing** (via the SpaCy library) to analyze grammatical structures, identifying key elements like nominal subjects (nsubj), passive subjects (nsubjpass), and direct objects (dobj). The system extracts focus phrases by parsing these elements and their dependency trees, isolating crucial tokens.
- **Answer Type Detection**: Anser detection utilises **rule-based classification** by mapping interrogative words (e.g., who, what, when) to answer types (e.g., person, thing, time, location), simplifying the process for identifying the required answer category.
- **Query Formulation**: This step leverages **tokenization** to isolate key phrases and constructs a simpler, keyword-based query. The system removes unnecessary parts of the question to focus on the terms most likely to retrieve relevant information.

### 2. **Coreference Resolution**
Coreference resolution ensures that the system accurately links pronouns and noun phrases to their referents, improving text understanding:
- The system uses **SpaCy’s Coreferee library**, which identifies coreference chains with high accuracy. The method processes sentences to detect pronouns and noun phrases that refer to the same entity across multiple sentences.
- **Dependency parsing** and **Named Entity Recognition (NER)** are applied to identify appositional phrases (noun phrases that refer to the same entity), helping to resolve more complex noun phrase references.
- This process ensures that pronouns like "he" or "they" and noun phrases like "the leader" consistently refer to specific entities, thereby improving the semantic coherence of the text.

### 3. **Passage Retrieval**
In this phase, the system retrieves the most relevant article and passage using traditional IR and NLP methods:
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: TF-IDF quantifies word importance by measuring how frequently terms appear in a passage relative to the entire document set. This helps the system down-weight common words while emphasizing unique terms. The system uses **TfidfVectorizer** from the sklearn library to compute a matrix of TF-IDF scores.
- **Cosine Similarity**: After calculating TF-IDF, **cosine similarity** is applied to compare the query vector with sentence vectors. This technique measures the angle between vectors, determining how similar the query is to each sentence.
- **N-grams (1 to 3 words)**: By using n-grams, the system captures word sequences (like phrases) rather than just individual words, improving retrieval by considering context and word order.

### 4. **Answer Extraction**
The system applies feature-based NLP techniques to extract the most relevant answers from a passage:
- **Appositional Feature Extraction**: This method identifies noun phrases appositionally related to the focus phrase, meaning they provide additional information about it. **Dependency parsing** is used to detect these relationships, ensuring the extracted answers are closely tied to the question focus.
- **NER (Named Entity Recognition)**: After extracting potential answers, the system uses NER to match the answer types to predefined categories like "person," "location," or "time." This ensures that the extracted noun phrases are correctly classified, further refining the answer pool.
- **Proximity-based Filtering**: To maintain accuracy, the system restricts candidate answers to those within close proximity (within three tokens) to the focus phrase, filtering out unrelated text and ensuring the answer’s relevance.

Additionally a BERT **(Bidirectional Encoder Representations from Transformers)** model pre-trained on SQuAD is utilised in this project to compare the performance of the IR-based QA method. The key parts of the BERT model are:

- **Tokenization**: The question and context are tokenized using the BERT tokenizer, with special tokens added (`[CLS]`, `[SEP]`). Additionally, padding and truncation are used to ensure the input fits within 512 tokens.
- **Model Input**: Converts the tokenized input into **input IDs** and an **attention mask**. These inputs are then fed into the pre-trained model.
- **Logits Prediction**: The model generates **start logits** and **end logits**, predicting the start and end positions of the answer in the context.
- **Answer Extraction**: Identifies the tokens with the highest start and end scores, marking the predicted span of the answer, which is then converted back into a string.
- **Handling Unanswerable Questions**: If the model is unable to find a valid answer, the system returns None to indicate that no answer exists.

## Dataset
Two datasets were used in this project for development and testing purposes:

1. **Primary Dataset (`news_dataset.csv`)**: This dataset consists of 1,000 news articles, each with key information like article ID, author name, publication date, related topic, and the full article text (context). A random sample of 100 articles was extracted with a fixed random state for consistency. During preprocessing, encoding errors were discovered, such as "?" replacing characters like apostrophes (e.g., "didn?t" instead of "didn't"). These errors were corrected to ensure proper semantic interpretation. Additional preprocessing steps included fixing formatting issues (e.g., concatenated words, whitespace normalization), and retaining capitalizations to assist in Named Entity Recognition (NER).

2. **SQuAD 2.0 Development Dataset (`dev-v2.0.json`)**: This dataset, sourced from the SQuAD website, contains 1,204 articles with associated question-answer pairs. Each article features multiple questions categorized as either answerable or impossible to answer. A random sample of 10 articles, including 115 question-answer pairs (69 of which were impossible to answer), was used for development. No preprocessing was necessary for this dataset as it was already prepared for use.

## Technology used:
The main Python technologies used in this project were:
- Pandas
- NumPy
- Json
- NLTK
- Re
- SpaCy
- Coreferee
- Matplotlib
- Seaborn
- Sklearn
- Transformers
- PyTorch
