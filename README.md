# Twords
Twitter Word Frequency Analysis

# Overview
Twords is a python class for collecting and analyzing tweets. Twords uses the java version of GetOldTweets (available [here](https://github.com/Jefferson-Henrique/GetOldTweets-java), which gets around the limitations of the Twitter API by querying the Twitter website directly. Users can collect ~3000 tweets per minute satisfying a particular search query, perform standard cleaning procedures, and visualize the relative rates certain words appear in tweets satisfying that query, all within Twords.




Twords takes in a csv file of tweets (gathered using GetOldTweets here: https://github.com/Jefferson-Henrique/GetOldTweets-java), 
cleans them, removes stopwords, and visualizes the frequency of words (including frequencies relative to background word frequencies drawn from a sample of tweets from the Twitter API). 

It also lets you add your own stop words and query specific words or groups of words for their relative frequencies. 

Ideas for things to predict: stock prices (try obscure companies), election results (try obscure elections), referendums votes (lots in california), how much money movies make, whether a startup fails (probaby best to focus on things that can be reviewed, since stanford nlp sentiment learner is trained on reviews). Can also try a few other sentiment libraries, like the facebook Fast Text one - I would need to train my own sentiment classifier for this one. Can also look at google open data sets. Finished sentiment analysis on stock tweets for netflix and amazon. 

<b> python dependencies:</b> numpy, pandas, nltk, seaborn, matplotlib, scipy, tailer, twitter-text-python
(May need to use nltk downloader to get stop words and punkt tokenizer model.)

Need Java sdk to run jar files as well

