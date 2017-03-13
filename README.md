# Twords

Twitter Word Frequency Analysis

Twords takes in a csv file of tweets (gathered using GetOldTweets here: https://github.com/Jefferson-Henrique/GetOldTweets-java), 
cleans them, removes stopwords, and visualizes the frequency of words (including frequencies relative to background word frequencies drawn from a sample of tweets from the Twitter API). 

It also lets you add your own stop words and query specific words or groups of words for their relative frequencies. 

Code now includes a separate sentiment class for returning the sentiment of the tweets as measured by Stanford CoreNLP code.

To do: add example comparing sentiment time series with stock price time series. Stock prices can be downloaded automatically with pandas. 

Also to do: get sentiment information from collection of background tweets, tho since training set is based on reviews might not be accurate. 

Ideas for things to predict: stock prices (try obscure companies), election results (try obscure elections), referendums votes (lots in california), how much money movies make, whether a startup fails (probaby best to focus on things that can be reviewed, since stanford nlp sentiment learner is trained on reviews). Can also try a few other sentiment libraries, like the facebook Fast Text one - I would need to train my own sentiment classifier for this one. Can also look at google open data sets. Finished sentiment analysis on stock tweets for netflix and amazon. 

<b> python dependencies:</b> numpy, pandas, nltk, seaborn, matplotlib, scipy, tailer, twitter-text-python

