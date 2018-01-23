# Extracting-keyphrases-from-a-scientific-article

This repository is created to explore the different ways to extract keyphrases from scientific articles. Currently it contains only baseline models for keyphrase extraction.

Dataset: semEval 2017 datset.

# Steps of approach:
1.tagging individual words using NLTK POS-tags.

2.using the above information to form valid phrases.

3.seiving out the potential candidate phrases from the list of valid phrases.

4.classifying the candidate phrases into keyphrases(1) or non-keyphrases(-1).

5.calculation

# Models:
1.	 RNN:
        Structure: embedding--->bi-directional lstm---> concat output---> average--->softmax

2.	CNN:
        Structure: embedding--->convolution layer--->max pooling--->fully connected layer--->softmax

3.	RCNN:
        Structure: embedding--->recurrent structure(convolution layer)--->max pooling--->fully connected layer--->softmax
        Representation of each word in key-phrase= [left_side_context_vector,word_embedding,right_side_context_vector]

# Performance:
| Models: | RNN	| CNN	|  RCNN |
|---|---|---|---|
|Precision| 0.3311 | 0.3001 | 0.3289 |
|Recall	 | 0.2169 | 0.2126 | 0.1910   | 
|f-score	 | 0.2621 |	0.2489 | 0.24166 |



