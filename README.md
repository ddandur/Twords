# Twords
Fast Twitter Dataset Creation and Twitter Word Frequency Analysis

# Overview
Twords is a python class for collecting tweets and investigating their word frequencies in an IPython notebook. Twords uses the java version of GetOldTweets by Jefferson Henrique (available [here](https://github.com/Jefferson-Henrique/GetOldTweets-java)) to download tweets, which gets around the limitations of the Twitter API by querying the Twitter website directly. The collection rate is about 3000 tweets per minute.

Once tweets are collected, Twords can be used to load tweets into a pandas dataframe, clean them, calculate their word frequencies, and visualize the relative frequency rates of words in the tweets as compared with the general Twitter background word frequency rates. Twords also provides functions that help in removing tweets that are uninteresting or spammy from the dataset.

# Motivation

I wanted to analyze big data sets of tweets containing certain search terms (like the word "charisma") but couldn't find an easy way to get them (for free). The Twitter API allows you to stream tweets by search term, but uncommon terms like "charisma" yielded only one or two tweets per minute.

The GetOldTweets library allows collecting from the Twitter website to get around this problem, but because it is querying the website directly, it would often encounter an error or hangup when gathering more than ~10,000 tweets. Twords take a search query and splits it into a series of smaller calls to GetOldTweets to avoid hangups on the Twitter website.

I also quickly discovered that Twitter is full of repetitive or spammy posts that clutter results for virtually any search. The filtering functions in Twords are meant to help with this. (For example, in late 2016 I streamed several GB tweets from the Twitter API using the stream functionality that is supposed to give a small percentage of the firehose - I don't know how Twitter decides which tweets get included in this "random" sample, but an incredible 2% of them were spam for The Weather Channel.)

Finally, I also used Twords to collect a large dataset of random tweets from Twitter in an attempt to get the background frequency rates for English words on Twitter. I did this by searching on 50 of the most common English words using Twords and validating by comparing to a large sample taken from the stream off the Twitter API (after removing the spam from the weather channel). These background freqeuncy rates are included with Twords as a baseline for comparing word frequencies. 

# Examples

## Tweet Collection

To collect 10,000 tweets containing the word "charisma", with 500 tweets collected per call to the java GetOldTweets jar file, open an IPython notebook in the root Twords directory and enter this:

```python
from twords.twords import Twords 

twit = Twords()
twit.jar_folder_path = "jar_files_and_background"
twit.create_java_tweets(total_num_tweets=10000, tweets_per_run=500, querysearch="charisma")
```

Collection proceeds at 2000-3000 tweets per minute. ~4 million tweets can be collected on a single machine in 24 hours.

To collect all tweets from Barack Obama's twitter account, enter this: 

```python
twit = Twords()
twit.jar_folder_path = "jar_files_and_background"
twit.get_all_user_tweets("barackobama", tweets_per_run=500)
```

In both cases the output will be a folder of csv files in your current directory that contains the searched tweets. 

## Tweet Cleaning

Once this folder of csv files exists, the data can be loaded into a pandas dataframe in Twords like this: 

```python
twit.data_path = "path_to_folder_containing_csv_files"
twit.get_java_tweets_from_csv_list()
```

The raw twitter data are now stored in the dataframe `twit.tweets_df`:

|  | username | date |retweets | favorites | text | mentions | hashtags | id | permalink
| ------------- |-------------:| -----:|------|----|-------|-----|------|----|----
|0|BarackObama|2007-04-29|786|429|Thinking we're only one signature away from ending the war in Iraq. Learn more at http://www.barackobama.com |NaN|NaN|44240662|https://twitter.com/BarackObama/status/44240662|
|1|BarackObama|2007-05-01|269|240|Wondering why, four years after President Bush landed on an aircraft carrier and declared ‘Mission Accomplished,’ we are still at war?|NaN|NaN|46195712|https://twitter.com/BarackObama/status/46195712|
|2|BarackObama|2007-05-07|6|4|At the Detroit Economic Club – Talking about the need to reduce our dependence on foreign oil.|NaN|NaN|53427172|https://twitter.com/BarackObama/status/53427172|


Now you can apply a variety of cleaning functions to the tweets to get them into a better form for word frequency analysis. Most of these are just convenience wrappers for standard manipulations in a pandas dataframe: 

``` python
twit.lower_tweets()
twit.remove_urls_from_tweets()
twit.remove_punctuation_from_tweets()
twit.convert_tweet_dates_to_standard()
twit.drop_duplicates_in_text()
twit.sort_tweets_by_date()
```

The cleaned tweets (still in the `text` column) now look like this: 

|  | username | date |retweets | favorites |     text      | mentions | hashtags | id | permalink
| ------------- |:-------------:| -----:|------|--------------|-------|-----|------|----|----
|0|BarackObama|2007-04-29|786|429|thinking were only one signature away from ending the war in iraq learn more at | NaN | NaN | 44240662 | https://twitter.com/BarackObama/status/44240662|
|1|BarackObama|2007-05-01|269|240|wondering why four years after president bush landed on an aircraft carrier and declared mission accomplished we are still at war|NaN|NaN|46195712|https://twitter.com/BarackObama/status/46195712|
|2|BarackObama|2007-05-07|6|4|at the detroit economic club talking about the need to reduce our dependence on foreign oil|NaN|NaN|53427172|https://twitter.com/BarackObama/status/53427172|

## Word Frequencies

Twords comes with a file containing background Twitter word frequencies for ~230,000 words (better called "tokens", since it includes things like "haha" and "hiiiii"). These are used as base rates to compare to when computing word frequencies from the Twords data set.

To load this background frequency information into a Twords instance: 

``` python
twit.background_path = 'path_to_background_csv_file'
twit.create_Background_dict()
```

Now we can create a new pandas dataframe `twit.word_freq_df` that will store the word frequencies of all the words in all tweets combined. To do this, a word bag list of all words combined is created and fed into an `nltk.FreqDist` object to compute word frequencies (not including frequencies of stop words, which are ignored), and the word frequencies for the most common `top_n_words` are stored in `twit.word_freq_df`. `twit.word_freq_df` also has columns that compare the computed word frequency to the background word frequency of each word.

``` python
twit.create_Stop_words()
twit.create_word_bag()
twit.make_nltk_object_from_word_bag()
twit.create_word_freq_df(top_n_words=10000)
```

A peek at `twit.word_freq_df` for Obama's twitter timeline:

|  | word | occurrences |frequency | relative frequency | log relative frequency | background_occur | 
| ------------- |:-------------:| -----:|------|--------------|-------|-----|
0	|sotu	|187	|0.001428|	9385.754002|	9.146948|	11
1	|middleclass	|161	|0.001229	|4938.256191	|8.504768	|18
2	|ofa	|321|	0.002451|	4663.818939|	8.447590	|38


With `twit.word_freq_df` in hand we can slice the data in many different ways and plot the results. Twords provides some convenience functions for quick plotting, and further exampels are included in the IPython notebooks in the examples folder.

As an example, here are the 10 words with highest relative frequency (that is, high frequency per word relative to background Twitter word rates) in Barack Obama's Twitter feed, where we require the background rate to be at least 6.5e-5:

![alt text](https://github.com/ddandur/Twords/blob/master/images/obama_top_10.png)

And here are the 10 words with lowest relative frequency with same background rate requirement: 

![alt text](https://github.com/ddandur/Twords/blob/master/images/obama_bottom_10.png)


