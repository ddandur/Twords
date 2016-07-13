# helper functions for twitter data analysis

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from operator import itemgetter
import seaborn as sns
import matplotlib.pyplot as plt
import qgrid
from math import log
import got
import time
import timeit


class Twords(object):
    """ Object that takes in tweets from Java twitter search engine and allows
    manipulation, analysis and visualization.
    """

    def __init__(self):
        self.data_path = ''
        self.background_path = ''
        self.background_dict = {}
        self.search_terms = []
        self.tweets_df = pd.DataFrame()
        self.word_bag = []
        self.freq_dist = nltk.FreqDist(self.word_bag)
        self.word_freq_df = pd.DataFrame()

    def __repr__(self):
        return "Twitter word analysis object"

    #############################################################
    # Methods to set attributes
    #############################################################

    def set_Data_path(self, data_path):
        """ data_path is path to data set from java twitter search"""
        self.data_path = data_path

    def set_Background_path(self, background_path):
        """ background_path is path to background data set from java
        twitter search"""
        self.background_path = background_path

    def set_Search_terms(self, search_terms):
        """ search_terms is a list of strings that were used in twitter search
        to obtain data in tweets_df
        """
        assert type(search_terms) == list
        for term in search_terms:
            assert type(term) == str
            assert term
        self.search_terms = search_terms

    def create_Background_dict(self):
        """ Create the dictionary of background word rates from file in the
        background data path
        """
        sample_rates = pd.read_csv(self.background_path, sep=",")
        self.background_dict = sample_rates[["word", "frequency"]].set_index("word")["frequency"].to_dict()


    #############################################################
    # Methods to gather tweets with Python GetOldTweets
    #############################################################

    def collect_one_run(self, search_terms, end_date, call_size):
        """ Does one twitter search with GetOldTweets python library.
        search_terms: string of terms to search on
        end_date: the date at which search should end (e.g., if you want
                    the most recent tweets, set this to today)
        call_size: number of tweets to return.

        Returns a list of tweets (packaged as dictionaries) and the most recent
        date that was searched. The most recent date is used when making
        repeated calls with the python library, which is necessary because
        searches of more than around 50,000 tweets get hung up on website.
        """
        tweetCriteria = got.manager.TweetCriteria()
        tweetCriteria.setQuerySearch(search_terms)
        tweetCriteria.setUntil(end_date)
        tweetCriteria.setMaxTweets(call_size)
        tweets = got.manager.TweetManager.getTweets(tweetCriteria)

        if len(tweets) == 0: # catches cases when twitter blocks search
            print "Could not retrieve any tweets"
            return None

        rows_list = []
        for tweet in tweets:
            row_dict = {"username": tweet.username,
                        "date": str(tweet.date.date()),
                        "retweets": tweet.retweets,
                        "favorites": tweet.favorites,
                        "text": tweet.text,
                        "mentions": tweet.mentions,
                        "hashtags": tweet.hashtags,
                        "id": tweet.id,
                        "permalink": tweet.permalink}
            rows_list.append(row_dict)
        return rows_list, row_dict["date"]

    def create_tweets_dataframe(self, search_terms, final_end_date,
                                num_tweets, call_size):
        """ Performs repeated calls to collect_one_run to get num_tweets into
        a dataframe. Each call to collect_one_run takes call_size.

        call_size should be on the order of several thousand for best results

        Each time a new call to collect_one_run is made, the date is
        incremented backward by one day. This means that if seven calls are
        made and each call only takes tweets from one day, seven different
        days (starting with final_end_date and moving backward in time) will
        be sampled.

        final_end_date should be a string in form "2015-12-31"
        """
        total_row_list = []
        search_date = final_end_date
        num_tweets_searched = 0

        starttime = time.time()

        while num_tweets_searched < num_tweets:
            row_list, last_date = self.collect_one_run(search_terms,
                                                       search_date,
                                                       call_size)

            total_row_list += row_list
            search_date = last_date
            num_tweets_searched += call_size

        # once all tweets are collected, combine them all into dataframe
        column_names = ["username",
                        "date",
                        "retweets",
                        "favorites",
                        "text",
                        "mentions",
                        "hashtags",
                        "id",
                        "permalink"]

        tweets_df = pd.DataFrame(total_row_list, columns = column_names)

        print "Time to collect ",  str(num_tweets), " tweets: ", time.time() - starttime, "seconds"

        self.tweets_df = tweets_df

    def save_tweets_df_to_csv(self, output_file_string):
        """ To save the dataframe to a csv file, use the pandas method without
        the index. output_file_string is name of output file
        """
        self.tweets_df.to_csv(output_file_string, index=False)

    #############################################################
    # Methods to gather tweets and prune (done every time)
    #############################################################

    def get_tweets(self):
        """ Takes path to twitter data obtained with java tweet search library
        and returns a dataframe of the tweets and their accompanying
        information. Dataframe has columns for username, date, retweets,
        favorites, text, mentions, and hashtag. The dataframe is stored under
        the attribute tweets_pd.
        """
        # Read in csv file with many columns to account for people who put many
        # semicolons in tweets, then keep only the rows that don't have
        # semicolons in a tweet by dropping rows with too many columns.
        # (Semicolons are the delimeter in the java twitter search library.)
        tweets = pd.read_csv(self.data_path, sep=";",
                             names=list('abcdefghijklmno'))
        tweets = tweets[tweets.k.isnull()]

        # Rename the columns with correct labels and drop row that is just
        # column names (this will index dataframe starting at 1).
        tweets.columns = tweets.iloc[0]
        tweets.drop(0, inplace=True)

        # Drop the extra columns on the end
        tweets = tweets[["username", "date", "retweets", "favorites", "text",
                         "mentions", "hashtags", "id", "permalink"]]

        # Reindex dataframe
        tweets.index = range(len(tweets))
        self.tweets_df = tweets


    def get_tweets_from_twitter_api_csv(self):
        """ Takes path to csv gathered with Twitter's API and returns the
        tweets_df dataframe. The number of columns might vary depending on how
        much information was taken from each tweet, but in order to be
        compatible with rest of code tweets should contain at least the text,
        mentions, usernames and hashtags.
        """


    def lower_tweets(self):
        """ Lowers case of text in all the tweets, usernames, and mentions in
        the tweets_df dataframe
        """
        self.tweets_df["username"] = self.tweets_df.username.str.lower()
        self.tweets_df["text"] = self.tweets_df.text.str.lower()
        self.tweets_df["mentions"] = self.tweets_df.mentions.str.lower()

    def drop_tweets_without_terms(self, tweet_list, term_list):
        """ Takes list of tweets and list of terms and drops the tweets that do
        NOT contain at least one of the terms.

        This is mostly as a sanity
        check. Cases where this might matter is if there is a mention or
        something about twitter I don't know that causes unwanted tweets to be
        included.
        """
        """
        # select only the tweets that contain the word "charisma" in the tweet itself
        tweets = tweets[tweets.text.str.contains("charisma") == True]
        # drop the tweets that contain "charisma" in a mention
        tweets = tweets.drop(tweets[tweets.mentions.str.contains("charisma") == True].index)
        # drop the tweets that contain the word "carpenter", since that probably refers to the actress
        tweets = tweets.drop(tweets[tweets.text.str.contains("carpenter") == True].index)
        # extract all tweets and convert into a python list of strings, with each string a separate tweet
        tweets_list = tweets["text"].tolist()
        """

        return self

    def drop_by_search_in_name(self):
        """ Drop tweets that contain element from search_terms in either
        username or mention (i.e., tweets where the search term in contained in
        twitter handle of someone writing or mentioned in tweet). Default
        values of terms list is search_terms attribute, but user can input
        arbitrary lists of strings to drop.
        """
        if not self.search_terms:
            print "search_terms is empty - add at least one term to " + \
                    "search_terms attribute"
            return self
        for term in self.search_terms:
            assert type(term) == str
            assert term # to make sure string isn't empty

        # Drop the tweets that contain any of search terms in either a username
        # or a mention
        for term in self.search_terms:
            mentions_index = self.tweets_df[self.tweets_df.mentions.str.contains(term) == True].index
            self.tweets_df.drop(mentions_index, inplace=True)
            username_index = self.tweets_df[self.tweets_df.username.str.contains(term) == True].index
            self.tweets_df.drop(username_index, inplace=True)

        # Reindex dataframe
        self.tweets_df.index = range(len(self.tweets_df))

    #############################################################
    # Methods to prune tweets after visual inspection
    #############################################################

    def drop_by_term_in_name(self, terms):
        """ Drop tweets that contain element from terms in either username or
        mention. The terms parameter must be a list of strings.

        This method is the same as drop_by_search_in_name method, except it
        takes arbitrary input from user.

        This can be used to help get rid of spam.
        """
        if not terms:
            print "terms is empty - enter at least one search terms string"
            return self
        for term in terms:
            assert type(term) == str
            assert term

        # Drop the tweets that contain any of terms in either a username
        # or a mention
        for term in terms:
            mentions_index = self.tweets_df[self.tweets_df.mentions.str.contains(term) == True].index
            username_index = self.tweets_df[self.tweets_df.username.str.contains(term) == True].index
            self.tweets_df.drop(mentions_index, inplace=True)
            self.tweets_df.drop(username_index, inplace=True)

        # Reindex dataframe
        self.tweets_df.index = range(len(self.tweets_df))

    def drop_by_term_in_tweet(self, terms):
        """ Drop tweets that contain element from terms in the tweet text.

        This is most useful for getting rid of repetitive or spammy tweets that
        appear to be distorting data.
        """
        if not terms:
            print "terms is empty - enter at least one term string"
            return self
        for term in terms:
            assert type(term) == str
            assert term

        # Drop the tweets that contain any of terms in text of tweet
        for term in terms:
            text_index = self.tweets_df[self.tweets_df.text.str.contains(term) == True].index
            self.tweets_df.drop(text_index, inplace=True)

        # Reindex dataframe
        self.tweets_df.index = range(len(self.tweets_df))

    #############################################################
    # Methods to do analysis on all tweets in bag-of-words
    #############################################################

    def create_word_bag(self):
        """ Takes tweet dataframe and outputs word_bag, which is a list of all
        words in all tweets, with punctuation and stop words removed.
        """
        # Convert dataframe tweets column to python list of tweets, then join
        # this list together into one long list of words
        tweets_list = self.tweets_df["text"].tolist()
        words_list = " ".join([str(i) for i in tweets_list])
        words_list = words_list.decode('utf-8')

        # Make list of stop words and punctuation to remove from list
        punctuation = list(string.punctuation)
        stop = stopwords.words('english') + punctuation + \
                ['rt', 'RT', 'via', 'http', "n't", "'s", "...", "''", "'m",
                    "--", "'ll", "'ve", "'re", "//www"]

        # Use nltk word tokenization to break list into words and remove
        # stop words
        tokens = nltk.word_tokenize(words_list)
        self.word_bag = [word for word in tokens if word not in stop]

    def make_nltk_object_from_word_bag(self, word_bag):
        """ Creates nltk word statistical object from the current word_bag
        attribute. word_bag is left as an input in case the user wants to
        create an nltk object with an external word bag.

        The most common method we'll use from this object is the
        frequency method, i.e. freq_dist.freq(term), where term is word in
        word bag.

        Use print(freq_dist) to get the number of unique words in corpus, as
        well as total number of words in corpus.

        Can use freq_dist.most_common(50) to get list of 50 most common words
        and the number of times each of them appears in text.
        """
        self.freq_dist = nltk.FreqDist(self.word_bag)


    #############################################################
    # Methods for investigating word frequencies
    #############################################################
    """ The frequencies_of_top_n_words method is used to create a dataframe
    that gives the word occurrences and word frequencies of the top n words in
    the corpus. This is created using the existing nltk object, and it is
    changed depending on how many words we wish to inspect graphically.

    The dataframe frequencies_of_top_n_words creates is stored in the
    word_freq_df attribute, which is a pandas dataframe. This is the dataframe
    that gets used in the plot_word_frequencies plotting function.

    For now the background corpus is derived from ~2.6 GB of twitter data,
    composing about 72 million words. The word frequency rates from this
    sample are stored in a frequency sample file that is then converted into
    a python dictionary for fast lookup.
    """

    def top_word_frequency_dataframe(self, n):
        """ Returns pandas dataframe of the most common n words in corpus,
        how often each of them occurred, their frequency in the corpus,
        relative frequency to background rates, and
        the log of the relative frequency to background rates.
        (The log is useful because, for example, a rate two times as high as
        background has log ratio of +x, and a rate two times lower than
        background rates has a log ration of -x.)

        n is the number of words we want to see. These words are draw in order
        of how frequently they are found in the corpus, so a large number of
        words should be chosen to make sure we find the interesting ones that
        appear much more often than in background corpus. (If a word appears
        often in our search corpus it may be because it also appear often in
        the background corpus, which is not of interest.)

        The actual words that were searched to collect the corpus are omitted
        from this dataframe (as long as self.search_terms has been set).
        """
        # make dataframe we'll use in seaborn plot
        num_words = n
        word_frequencies_list = []
        for word, occurrences in self.freq_dist.most_common(num_words):
            # determine whether word appears in background dict; if it does
            # not, the frequency ratio is set to zero
            if word in self.search_terms:
                continue
            if word in self.background_dict.keys():
                freq_ratio = self.freq_dist.freq(word)/self.background_dict[word]
                background_freq = self.background_dict[word]
                log_freq_ratio = log(freq_ratio)
            else:
                freq_ratio = 0
                background_freq = 0
                log_freq_ratio = 0

            word_frequencies_list.append((word, occurrences,
                                          self.freq_dist.freq(word),
                                          freq_ratio, log_freq_ratio))
        word_freq_df = pd.DataFrame(word_frequencies_list,
                                columns=['word', 'occurrences', 'frequency',
                                'relative frequency', 'log relative frequency'])
        self.word_freq_df = word_freq_df

    def plot_word_frequencies(self, plot_string):
        """ Plots of given value about word, where plot_string is a string
        that gives quantity to be plotted

        plot_string must be a column of word_freq_df dataframe, i.e. must be
        "occurrences", "frequency", "relative frequency", or
        "log relative frequency".
        """

        num_words = len(self.word_freq_df)
        try:
            self.word_freq_df.set_index("word")[plot_string].plot.barh(figsize=(20,
                num_words/2.), fontsize=30, color="c"); plt.title(plot_string, fontsize=30)
        except:
            raise Exception("Input string must be column name of word_freq_df")


        """ This was more customized code that can be used later if needed - for
        now the pandas default plotting code is good enough for most purposes


        sns.set(style="darkgrid")
        num_words = len(self.word_freq_df)
        # Initialize the matplotlib figure - the second number in figure gives
        # height, this will need to depend on how many words are included in
        # figure
        f, ax = plt.subplots(figsize=(16, num_words/2.))
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)

        # Plot the frequencies
        sns.set_color_codes("pastel")
        sns.barplot(x=plot_string, y="word", data=self.word_freq_df,
                    label="frequency", color="b")

        # Add informative axis label
        max_value = self.word_freq_df.iloc[0].frequency # find maximum frequency
        # adjust axis to be slightly larger than this max frequency
        ax.set(xlim=(0, max_value*1.1), ylabel="", xlabel="Word frequency")
        ax.set_xlabel(plot_string, fontsize=30)
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.tick_params(axis='x', labelsize=20) # size of numerical labels
        """



    #############################################################
    # Methods to inspect tweets in tweets_df dataframe
    #############################################################
    """ These methods are used to inspect tweets of interest in the main
    dataframe tweets_df. A typical workflow is to visualize tweet word
    frequencies using visualization functions, then inspect a sample of tweets
    that contain a word of interest. If these tweets appear to be unwanted they
    can then be dropped using the dropping functions above.

    Note about displaying tweets in pandas in readable form: need to set
    pd.set_option('display.max_colwidth', -1) and/or
    pd.set_option('display.width',800)

    This makes it so entire tweet is displayed without cutoff when only tweets
    are presented in dataframe.

    (Can also use qgrid.show_grid to get a more excel-like out with scrolling
    and column resizing, which can be done later. For now just use pandas
    output.)

    Can enter pd.describe_option('display') to get comprehensive list of
    settings for ipython displays.
    """

    def tweets_containing(self, term, qg=False):
        """ Returns all tweets that contain term from tweets_df.

        The returned object is a dataframe that contains the rows of tweets_df
        dataframe that have tweets containing term.

        Depending on how long this function takes to run, next iteration of
        this function should take a sample instead of returning all tweets.
        """
        assert type(term) == str
        assert term

        pd.set_option('display.max_colwidth', -1)
        tweets_containing = self.tweets_df[self.tweets_df.text.str.contains(term) == True]
        if qg:
            qgrid.nbinstall()
            qgrid.show_grid(tweets_containing[["text", "username", "date", "mentions"]],
            remote_js=False, show_toolbar=True, grid_options={'forceFitColumns': False,
            'defaultColumnWidth': 100})
        else:
            return tweets_containing[["username", "text"]]


    def tweets_by(self, username, qg=False):
        """ Returns all tweets by username from tweets_df.

        Similar to above function except searches by username rather than
        tweet text.
        """
        assert type(username) == str
        assert username

        pd.set_option('display.max_colwidth', -1)
        tweets_by = self.tweets_df[self.tweets_df.username == username]
        if qg:
            qgrid.nbinstall()
            qgrid.show_grid(tweets_by[["text", "username", "date", "mentions"]],
            remote_js=False, show_toolbar=True, grid_options={'forceFitColumns': False,
            'defaultColumnWidth': 100})
        else:
            return tweets_by[["username", "text"]]
