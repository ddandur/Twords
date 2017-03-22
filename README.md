# Twords
Twitter Word Frequency Analysis

# Description
Twords is a python class for collecting and analyzing tweets from the Twitter website. Twords uses the java 




Twords takes in a csv file of tweets (gathered using GetOldTweets here: https://github.com/Jefferson-Henrique/GetOldTweets-java), 
cleans them, removes stopwords, and visualizes the frequency of words (including frequencies relative to background word frequencies drawn from a sample of tweets from the Twitter API). 

It also lets you add your own stop words and query specific words or groups of words for their relative frequencies. 

Ideas for things to predict: stock prices (try obscure companies), election results (try obscure elections), referendums votes (lots in california), how much money movies make, whether a startup fails (probaby best to focus on things that can be reviewed, since stanford nlp sentiment learner is trained on reviews). Can also try a few other sentiment libraries, like the facebook Fast Text one - I would need to train my own sentiment classifier for this one. Can also look at google open data sets. Finished sentiment analysis on stock tweets for netflix and amazon. 

<b> python dependencies:</b> numpy, pandas, nltk, seaborn, matplotlib, scipy, tailer, twitter-text-python
(May need to use nltk downloader to get stop words and punkt tokenizer model.)

Need Java sdk to run jar files as well

# License 

MIT License

Copyright (c) [2017] [Daniel Dandurand]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
