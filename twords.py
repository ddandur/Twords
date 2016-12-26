#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from operator import itemgetter
import seaborn as sns
import matplotlib.pyplot as plt
import qgrid
from math import log, ceil, sqrt
import time
import timeit
import datetime
import tailer
import subprocess
from os import listdir
from os.path import join as pathjoin
import re
import scipy.stats as st
from ttp import ttp

import os, sys, inspect

# use this if you want to include modules from a subfolder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"GetOldTweets-python")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import got

pd.set_option('display.max_colwidth', -1)


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
        self.stop_words = []
        self.match_urls = re.compile(r"""(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))""")

    def __repr__(self):
        return "Twitter word analysis object"

    #############################################################
    # Methods to set attributes
    #############################################################

    def set_Data_path(self, data_path):
        """ data_path (string) is path to data set from java twitter search.
        It can be either path to single file, or path to directory
        containing several java csv files."""
        self.data_path = data_path

    def set_Background_path(self, background_path):
        """ background_path (string) is path to background data
        Form of background data file is csv with columns 'word', 'occurrences',
        and 'frequency' for words as they occur in some background corpus.
        """
        self.background_path = background_path

    def set_Search_terms(self, search_terms):
        """ search_terms is a list of strings that were used in twitter search
        to obtain data in tweets_df.

        The strings will be converted to unicode inside Twords, even though the
        user may enter them as ordinary strings.
        """
        assert type(search_terms) == list
        for term in search_terms:
            assert type(term) in (str, unicode)
        unicode_list = [x.decode("utf-8") if type(x) == str
                        else x for x in search_terms]
        self.search_terms = unicode_list

    def create_Background_dict(self):
        """ Create the dictionary of background word rates from file in the
        background data path.
        key: word (string)
        value: tuple of form (frequency, occurrences), where
               frequency is frequency of word in background data set, and
               occurrences is total number of occurrences in background data
               set
        """
        sample_rates = pd.read_csv(self.background_path, sep=",", encoding='utf-8')
        background_dict = dict(zip(sample_rates["word"], zip(sample_rates["frequency"],sample_rates["occurrences"])))
        self.background_dict = background_dict

    def create_Stop_words(self):
        """ Create list of stop words used in create_word_bag function.
        Stop words created here are defaults - the user may add new stop words
        later with the add_stop_word function.
        """
        punctuation = [item.decode('utf-8') for item in list(string.punctuation)]
        stop = stopwords.words('english') + punctuation + \
               [u'rt', u'RT', u'via', u'http', u"n't", u"'s", u"...", u"''",
                u"'m", u"--", u"'ll", u"'ve", u"'re", u"//www"]
        self.stop_words = stop

    #############################################################
    # Methods to gather tweets via keyword search with
    # Python GetOldTweets
    # Note: not supported any longer, use java version
    #############################################################

    def collect_one_run(self, search_terms, end_date, call_size):
        """ Does one twitter search with GetOldTweets python library.

        search_terms: (string) string of terms to search on
        end_date: (string) the date at which search should end (e.g., if you
                    want the most recent tweets, set this to today). String
                    should be of the form "2015-12-31".
        call_size: (int) number of tweets to return.

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

        if len(tweets) == 0:  # catches cases when twitter blocks search
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

        Each time a new call to collect_one_run is made, the date is
        incremented backward by one day. This means that if seven calls are
        made and each call only takes tweets from one day, seven different
        days (starting with final_end_date and moving backward in time) will
        be sampled.

        final_end_date (string): data of form "2015-12-31"
        call_size (int): number of tweets to collect on one call, should be on
                        order of several thousand for best results
        search terms (string): string of terms to search on
        num_tweets (int): total number of tweets to return. The number of calls
                          made to collect_one_run will be num_tweets/call_size
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

        tweets_df = pd.DataFrame(total_row_list, columns=column_names)

        print "Time to collect ",  str(num_tweets), " tweets: ", (time.time() \
              - starttime)/60., "minutes"

        self.tweets_df = tweets_df

    def save_tweets_df_to_csv(self, output_file_string):
        """ To save the dataframe to a csv file, use the pandas method without
        the index.

        output_file_string (string): name of output file
        """
        self.tweets_df.to_csv(output_file_string, index=False)

    ##############################################################
    # Methods to gather tweets via keyword search with
    # Java GetOldTweets
    ##############################################################

    def get_tweets_from_single_java_csv(self):
        """ Takes path to twitter data obtained with java tweet search library
        and builds a dataframe of the tweets and their accompanying
        information. Dataframe has columns for username, date, retweets,
        favorites, text, mentions, and hashtag. The dataframe is stored under
        the attribute tweets_pd.
        """
        # Read in csv file with many columns to account for people who put many
        # semicolons in tweets, then keep only the rows that don't have
        # semicolons in a tweet by dropping rows with too many columns.
        # (Semicolons are the delimeter in the java twitter search library.)
        tweets = pd.read_csv(self.data_path, sep=";",
                             names=list('abcdefghijklmno'), encoding='utf-8')
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

    def validate_date(self, date_text):
        """ Return true if date_text is string of form '2015-06-29',
        false otherwise.

        date_text (string): date
        """
        try:
            datetime.datetime.strptime(date_text, '%Y-%m-%d')
            return True
        except ValueError:
            return False

    def convert_date_to_standard(self, date_text):
        """ Convert a date string of form u"yyyy/mm/dd" into form u"yyyy-mm-dd"
        for use with the python date module.
        """
        assert type(date_text) in (str, unicode)
        date_text = date_text.replace('/', '-')
        return date_text

    def create_java_tweets(self, total_num_tweets, tweets_per_run, querysearch,
                           final_until=None, output_folder="output",
                           decay_factor=4, all_tweets=True):
        """ Function that calls java program iteratively further and further
        back in time until the desired number of tweets are collected. The
        "until" parameter gives the most recent date tweets can be found from,
        and the search function works backward in time progressively from that
        date until the max number of tweets are found. Thus each new call to
        get_one_java_run_and_return_last_line_date will start the search one
        day further in the past.

        total_num_tweets: (int) total number of tweets to collect

        tweets_per_run: (int) number of tweets in call to java program - should
                        not be over 50,000, better to keep around 10,000

        querysearch: (string) string defining query for twitter search - see
                     Henrique code
                     (e.g, "europe refugees" for search for tweets containing
                     BOTH "europe" and "refugees" - currently putting in OR by
                     hand does not yield desired result, so two separate
                     searches will have to be done for "OR" between words

        final_until: (string) date string of the form '2015-10-09' that gives
                     ending date that tweets are searched before (this is
                     distinguished from the changing "until" that is used in
                     the calls to get_one_java_run_and_return_last_line_date).
                     If left as "None" it defaults to the current date.

        output_folder: (string) name of folder to put output in

        decay_factor: (int) how quickly to wind down tweet search if errors
                      occur and no tweets are found in a run - a failed run
                      will count as tweets_per_run/decay_factor tweets found,
                      so the higher the factor the longer the program will try
                      to search for tweets even if it gathers none in a run

        all_tweets: (bool) flag for which jar to use - True means use
                    all_tweets jar, False means use top_tweets jar
        """

        if final_until is None:
            final_until = str(datetime.datetime.now())[:10]

        print "Collecting", str(total_num_tweets), "tweets with", \
              str(tweets_per_run), "tweets per run."
        print "Expecting", \
              str(int(ceil(total_num_tweets/float(tweets_per_run)))), \
              "total runs"
        start_time = time.time()

        tweets_searched = 0
        run_counter = 1
        # create folder that tweets will be saved into
        subprocess.call(['mkdir', output_folder])
        until = final_until

        while tweets_searched < total_num_tweets:
            print "Collecting run", run_counter
            run_counter += 1
            # call java program and get date of last tweet found
            last_date = self.get_one_java_run_and_return_last_line_date(
                                querysearch, until, tweets_per_run, all_tweets)
            # rename each output file and put into new folder - output file
            # is named by until date
            new_file_location = output_folder + '/' + querysearch + '_' + \
                                until + '.csv'
            subprocess.call(['mv', 'output_got.csv', new_file_location])
            # if last_date is usual date proceed as normal - if not raise error
            # and stop search
            if self.validate_date(last_date):
                until = last_date
                tweets_searched += tweets_per_run
            else:
                # set search date one day further in past
                new_until_date_object = datetime.datetime.strptime(until, '%Y-%m-%d') \
                                        - datetime.timedelta(days=1)
                until = str(new_until_date_object)[:10]
                # consider this a few tweets searched so program doesn't run
                # forever if it gathers no tweets
                tweets_searched += (tweets_per_run)/float(decay_factor)

        self.data_path = output_folder
        self.search_terms = querysearch.split()
        print "Total time to collect", str(total_num_tweets), "tweets:", \
              (time.time() - start_time)/60., "minutes"

    def get_one_java_run_and_return_last_line_date(self, querysearch, until,
                                                   maxtweets, all_tweets=True,
                                                   since=None,
                                                   return_line=True):
        """ Create one java csv using java jar (either Top Tweets or All tweets
        as specified in all_tweets tag) and return date string from last tweet
        collected.

        querysearch: (string) query string, usually one word - multiple words
                     imply an "AND" between them
        maxtweets: (int) number of tweets to return
        since: (string of form '2015-09-30') string of date to search since;
                this is optional and won't be used when using the
                create_java_tweets function
        until: (string of form '2015-09-30') string of date to search until,
               since search is conducted backwards in time
        return_line (bool): whether to return date from last line or not; if
                            true the date from the last line in the csv is
                            returned
        """

        start_time = time.time()

        # choose which jar file to use
        jar_string = 'got_top_tweets.jar'
        if all_tweets:
            jar_string = 'got_all_tweets.jar'

        # create search string
        quotation_mark = '"'
        query_string = 'querysearch=' + quotation_mark + querysearch + quotation_mark
        until_string = 'until=' + until
        maxtweets_string = 'maxtweets=' + str(maxtweets)

        # create output_got.csv file of tweets with these search parameters
        if since is None:
            subprocess.call(['java', '-jar', jar_string, query_string,
                             until_string, maxtweets_string])
        else:
            since_string = 'since=' + since
            subprocess.call(['java', '-jar', jar_string, query_string,
                             since_string, until_string, maxtweets_string])

        # find date on last tweet in this file (in last line of file)
        last_line = tailer.tail(open('output_got.csv'), 1)[0]
        date_position = last_line.find(';')
        date_string = last_line[date_position+1:date_position+11]
        date_string = self.convert_date_to_standard(date_string)

        print "Time to collect", str(maxtweets), "tweets:", \
              (time.time() - start_time)/60., "minutes"

        if return_line:
            return date_string

    def get_list_of_csv_files(self, directory_path):
        """ Return list of csv files inside a directory

        directory_path: (string) path to directory holding csv files of
        interest
        """
        return [pathjoin(directory_path, f) for f in listdir(directory_path)
                if f[-4:] == '.csv']

    def get_java_tweets_from_csv_list(self, list_of_csv_files=None):
        """ Create tweets_df from list of tweet csv files

        list_of_csv_files: python list of paths (the paths are strings) to csv
                           files containing tweets - if list_of_csv_files is
                           None then the files contained inside self.data_path
                           are used
        """
        if list_of_csv_files is None:
            list_of_csv_files = self.get_list_of_csv_files(self.data_path)
        path_dict = {}
        # create dictionary with paths for keys and corresponding tweets
        # dataframe for values
        for path in list_of_csv_files:
            tweets = pd.read_csv(path, sep=";", names=list('abcdefghijklmno'),
                                 encoding='utf-8')
            tweets = tweets[tweets.k.isnull()]
            tweets.columns = tweets.iloc[0]
            tweets.drop(0, inplace=True)
            tweets = tweets[["username", "date", "retweets", "favorites",
                            "text", "mentions", "hashtags", "id", "permalink"]]
            tweets.index = range(len(tweets))
            path_dict[path] = tweets

        # join all created dataframes together into final tweets_df dataframe
        self.tweets_df = pd.concat(path_dict.values(), ignore_index=True)

    ##############################################################
    # Methods to gather user timeline tweets with
    # Java GetOldTweets
    ##############################################################

    def get_user_tweets(self, user, max_tweets, start_date=None,
                         end_date=None, all_tweets=True, return_line=True):
        """ Returns max_tweets from Twitter timeline of user. Appears to work
        better when start and end dates are included. The returned tweets
        include tweets on the start_date, and up to (but not including) tweets
        on the end_date.

        If only an end_date is provided, then the tweets are searched backward
        in time starting at the end date and continuing until max_tweets
        have been found.

        user (string): Twitter handle of user, e.g. barackobama
        max_tweets (int): number of tweets to return for that user; set
                          max_tweets to -1 to return all tweets in timeline
        start_date (string): starting date for search of form "2015-09-30"
        end_date (string): ending date for search of form "2015-09-30"
        all_tweets (bool): whether to use "top_tweets" or "all_tweets" java
                           jar file
        return_line (bool): whether to return date on last tweet returned;
                            needed for function that makes repeated calls to
                            this function, e.g. get_all_user_tweets
        """

        start_time = time.time()

        # choose which jar file to use
        jar_string = 'got_top_tweets.jar'
        if all_tweets:
            jar_string = 'got_all_tweets.jar'

        # create search string
        quotation_mark = '"'
        user_string = 'username=' + user
        maxtweets_string = 'maxtweets=' + str(max_tweets)

        if start_date is not None:
            since_string = 'since=' + start_date
        if end_date is not None:
            until_string = 'until=' + end_date

        # create output_got.csv file of tweets with these search parameters
        if start_date is None and end_date is None:
            subprocess.call(['java', '-jar', jar_string, user_string,
                             maxtweets_string])
        elif start_date is None and end_date is not None:
            subprocess.call(['java', '-jar', jar_string, user_string,
                             until_string, maxtweets_string])
        else:
            subprocess.call(['java', '-jar', jar_string, user_string,
                             since_string, until_string, maxtweets_string])

        # find date on last tweet in this file (in last line of file)
        last_line = tailer.tail(open('output_got.csv'), 1)[0]
        date_position = last_line.find(';')
        date_string = last_line[date_position+1:date_position+11]
        date_string = self.convert_date_to_standard(date_string)

        print "Time to collect", str(max_tweets), "tweets:", \
              (time.time() - start_time)/60., "minutes"

        if return_line:
            return date_string

    def get_all_user_tweets(self, user, tweets_per_run):
        """ Return all tweets in a user's timeline. This is necessary
        to do in batches since one call to get_user_tweets does not return
        all of the tweets (too many in one run breaks the web-scrolling
        functionality of GetOldTweets). The tweets are saved as series of
        csv files into output folder named by username of twitter user.

        The final date is one day after the current date, since tweets are
        returned up to (but not including) the end_date in get_user_tweets
        function.

        This function will return duplicates of some tweets to be sure all
        tweets are obtained - these can be eliminated by simply dropping
        duplicates in the text column of resulting pandas dataframe.

        Function typically fails to return every single tweets, but captures
        most (~87% for barackobama) - best performance when tweets_per_run
        is around 500.

        user (string): twitter handle of user, e.g. "barackobama"
        tweets_per_run (int): how many tweets to pull in each run
        """
        # increment the date one day forward from returned day when calling
        # get_user_tweets to be sure all tweets in overlapping
        # range are returned - experimentation showed that tweets on the edge
        # between date runs can be lost otherwise

        start_time = time.time()
        print "Collecting tweets with", str(tweets_per_run), "tweets per run."

        # create folder that tweets will be saved into
        subprocess.call(['mkdir', user])

        # set one day in future so that all tweets up to today are returned;
        # necessary because tweets are returned on dates up to but not
        # including end date
        final_until = str(datetime.datetime.now() +
                          datetime.timedelta(days=1))[:10]
        until = final_until
        continue_search = True
        run_counter = 1

        while continue_search:
            print "Collecting run", run_counter
            run_counter += 1
            # call user function and get date of last tweet found
            last_date = self.get_user_tweets(user, tweets_per_run,
                                             end_date=until)
            # rename each output file and put into new folder - output file
            # is named by until date
            new_file_location = user + '/' + until + '.csv'
            subprocess.call(['mv', 'output_got.csv', new_file_location])

            # if last_date is a date proceed as normal - if the last_date
            # hasn't changed, raise comment below
            if self.validate_date(last_date):
                until_minus_day_object = datetime.datetime.strptime(until, '%Y-%m-%d') \
                                        - datetime.timedelta(days=1)
                until_minus_day = str(until_minus_day_object)[:10]
                if last_date == until_minus_day:
                    # from experimentation sometimes a query of many tweets
                    # will get "stuck" on a day long before 500 tweets have
                    # been reached - solution is just increment day as usual
                    print "Tweets timeline incremented by only one day - may \
                           need larger tweets_per_run, or could just be \
                           regular stutter in querying timeline."
                    until = last_date
                else:
                    # this increment is to avoid losing tweets at the edge
                    # between date queries - experimentation showed they can
                    # be lost without this redundancy - this means when tweets
                    # are read there may be duplicates that require deletion
                    new_until_date_object = datetime.datetime.strptime(last_date, '%Y-%m-%d') \
                                            + datetime.timedelta(days=1)
                    until = str(new_until_date_object)[:10]
            else:
                continue_search = False

        # set data path to new output folder to read in new tweets easily
        self.data_path = user
        print "Total time to collect tweets:", \
              (time.time() - start_time)/60., "minutes"


    #############################################################
    # Methods to gather tweets from Twitter API stream file
    #############################################################

    def get_tweets_from_twitter_api_csv(self, path_to_api_output=None):
        """ Takes path to csv gathered with Twitter's API and returns the
        tweets_df dataframe. The number of columns might vary depending on how
        much information was taken from each tweet, but it is good to include
        "username", "text", "mentions" and "hashtags", since those column
        names are referenced later, e.g. for cleaning and for dropping tweets
        by username if username contains a search term.

        In case the file does not have a header, the user can add a header
        manually to tweets_df after inspecting it.

        path_to_api_output: (string) path to csv of tweet information obtained
                            by using Twitter api directly
        """
        if path_to_api_output is None:
            path_to_api_output = self.data_path
        self.tweets_df = pd.read_csv(path_to_api_output, encoding='utf-8')

    #############################################################
    # Methods to clean and prune tweets
    #############################################################

    def lower_tweets(self):
        """ Lowers case of text in all the tweets, usernames, mentions and
        hashtags in the tweets_df dataframe, if the dataframe has those
        columns.
        """
        column_names = list(self.tweets_df.columns.values)
        if "username" in column_names:
            self.tweets_df["username"] = self.tweets_df.username.str.lower()
        if "text" in column_names:
            self.tweets_df["text"] = self.tweets_df.text.str.lower()
        if "mentions" in column_names:
            self.tweets_df["mentions"] = self.tweets_df.mentions.str.lower()
        if "hashtags" in column_names:
            self.tweets_df["hashtags"] = self.tweets_df.hashtags.str.lower()

    def keep_tweets_with_terms(self, term_list):
        """ Drops all the tweets in tweets_df that do NOT contain at least one
        term from term_list. This is useful for handling data from Twitter API
        search stream, where it is often easiest to collect a single big stream
        using several search terms and then parse the stream later.

        term_list (string or list of strings): collection of terms to drop on
        """
        if type(term_list) == str:
            assert len(term_list) > 0
            keep_index = self.tweets_df[self.tweets_df.text.str.contains(term_list) == True].index
            self.tweets_df = self.tweets_df.iloc[keep_index]

        if type(term_list) == list:
            keep_index = pd.core.index.Int64Index([], dtype='int64')
            for term in term_list:
                assert len(term) > 0
                term_keep_index = self.tweets_df[self.tweets_df.text.str.contains(term) == True].index
                keep_index = keep_index.append(term_keep_index)
            keep_index = keep_index.drop_duplicates()
            self.tweets_df = self.tweets_df.iloc[keep_index]
        # Reindex dataframe
        self.tweets_df.index = range(len(self.tweets_df))

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
            assert type(term) in (str, unicode)
            assert term  # to make sure string isn't empty

        # Drop the tweets that contain any of search terms in either a username
        # or a mention
        column_names = list(self.tweets_df.columns.values)
        for term in self.search_terms:
            if "mentions" in column_names:
                mentions_index = self.tweets_df[self.tweets_df.mentions.str.contains(term) == True].index
                self.tweets_df.drop(mentions_index, inplace=True)
            if "username" in column_names:
                username_index = self.tweets_df[self.tweets_df.username.str.contains(term) == True].index
                self.tweets_df.drop(username_index, inplace=True)

        # Reindex dataframe
        self.tweets_df.index = range(len(self.tweets_df))

    def drop_duplicates_in_text(self):
        """ Drop duplicate tweets in tweets_df (except for the first instance
        of each tweet)
        """
        self.tweets_df.drop_duplicates("text", inplace=True)
        # Reindex dataframe
        self.tweets_df.index = range(len(self.tweets_df))

    def keep_only_unicode_tweet_text(self):
        """ Keeps only tweets where tweet text is unicode. This drops the
        occasional tweet that has a NaN value in dataset, which becomes a float
        when read into tweets_df.
        """
        self.tweets_df["text_type"] = self.tweets_df["text"].map(lambda text: type(text))
        self.tweets_df = self.tweets_df[self.tweets_df.text_type == unicode]
        del self.tweets_df["text_type"]
        # Reindex dataframe
        self.tweets_df.index = range(len(self.tweets_df))

    def remove_urls_from_single_tweet(self, tweet):
        """ Remove urls from text of a single tweet.

        This uses python tweet parsing library that misses some tweets but
        doesn't get hung up with evil regex taking too long.
        """
        # this regex for matching urls is from stackoverflow:
        # http://stackoverflow.com/questions/520031/whats-the-cleanest-way-to-extract-urls-from-a-string-using-python
        # http://daringfireball.net/2010/07/improved_regex_for_matching_urls
        # library installed with: pip install twitter-text-python
        p = ttp.Parser()
        result = p.parse(tweet)
        for x in result.urls:
            tweet = tweet.replace(x, "")
        tweet = tweet.strip()
        return tweet

    def remove_urls_from_tweets(self):
        """ Remove urls from all tweets in self.tweets_df
        """
        self.tweets_df["text"] = self.tweets_df["text"].map(self.remove_urls_from_single_tweet)

    def convert_tweet_dates_to_standard(self):
        """ Convert tweet dates from form "yyyy/mm/dd" to "yyyy-mm-dd".
        """
        self.tweets_df["date"] = self.tweets_df["date"].map(self.convert_date_to_standard)

    def sort_tweets_by_date(self):
        """ Sort tweets by their date - useful for any sort of time series
        analysis, e.g. analyzing sentiment changes over time.
        """
        self.tweets_df.sort_values("date", inplace=True)
        # Reindex dataframe
        self.tweets_df.index = range(len(self.tweets_df))

    #############################################################
    # Methods to prune tweets after visual inspection
    #############################################################

    def drop_by_term_in_name(self, terms):
        """ Drop tweets that contain element from terms in either username or
        mention. The terms parameter must be a list of strings.

        This method is the same as drop_by_search_in_name method, except it
        takes arbitrary input from user. This can be used to help get rid of
        spam.

        terms (list): python list of strings
        """
        if not terms:
            print "terms is empty - enter at least one search terms string"
            return self
        for term in terms:
            assert type(term) in (str, unicode)
            assert term

        # Drop the tweets that contain any of terms in either a username
        # or a mention
        column_names = list(self.tweets_df.columns.values)
        for term in terms:
            if "mentions" in column_names:
                mentions_index = self.tweets_df[self.tweets_df.mentions.str.contains(term) == True].index
                self.tweets_df.drop(mentions_index, inplace=True)
            if "username" in column_names:
                username_index = self.tweets_df[self.tweets_df.username.str.contains(term) == True].index
                self.tweets_df.drop(username_index, inplace=True)

        # Reindex dataframe
        self.tweets_df.index = range(len(self.tweets_df))

    def drop_by_term_in_tweet(self, terms):
        """ Drop tweets that contain element from terms in the tweet text.
        Terms can be either a string (which is treated as one term) or a list
        of strings (which area each treated as a separate drop case).

        This is most useful for getting rid of repetitive or spammy tweets that
        appear to be distorting data.

        This is also useful for dropping retweets, which can be accomplished
        by dropping tweets containing the string "rt @"

        terms (string or python list of strings): terms that appear in tweets
                                                  we want to drop
        """
        if type(terms) in (str, unicode):
            text_index = self.tweets_df[self.tweets_df.text.str.contains(terms) == True].index
            self.tweets_df.drop(text_index, inplace=True)

        elif type(terms) == list:
            for term in terms:
                assert type(term) in (str, unicode)
                assert len(term) > 0
                text_index = self.tweets_df[self.tweets_df.text.str.contains(term) == True].index
                self.tweets_df.drop(text_index, inplace=True)

        else:
            raise Exception("Input must be string or list of string.")
        # Reindex dataframe
        self.tweets_df.index = range(len(self.tweets_df))

    def add_stop_word(self, stopwords):
        """ Add word or list of words to stop words used in create_word_bag.
        The word might be a url or spam tag. A common case is parts of urls
        that are parsed into words (e.g. from youtube) that appear repeatedly.

        stopwords: (string or list of strings):
        """
        if type(stopwords) in (str, unicode):
            if type(stopwords) == str:
                # convert string to unicode if not unicode already
                stopwords = stopwords.decode('utf-8')
            self.stop_words = self.stop_words + [stopwords]

        elif type(stopwords) == list:
            for term in stopwords:
                assert type(term) in (str, unicode)
                assert len(term) > 0
            unicode_terms_list = [term if type(term) == unicode
                                  else term.decode('utf-8')
                                  for term in stopwords]
            self.stop_words = self.stop_words + unicode_terms_list

        else:
            raise Exception("Input must be string or list of strings.")

    #############################################################
    # Methods to do analysis on all tweets in bag-of-words
    #############################################################

    def create_word_bag(self):
        """ Takes tweet dataframe and outputs word_bag, which is a list of all
        words in all tweets, with punctuation and stop words removed. word_bag
        is contained inside the attribute self.word_bag.

        This method will often be called repeatedly during data inspection, as
        it needs to be redone every time some tweets are dropped from
        tweets_df.
        """
        start_time = time.time()
        # Convert dataframe tweets column to python list of tweets, then join
        # this list together into one long list of words
        tweets_list = self.tweets_df["text"].tolist()

        words_string = " ".join(tweets_list)
        print "Time to make words_string: ", (time.time() - start_time)/60., "minutes"

        start_time = time.time()
        # Use nltk word tokenization to break list into words and remove
        # stop words
        tokens = nltk.word_tokenize(words_string)
        print "Time to tokenize: ", (time.time() - start_time)/60., "minutes"

        start_time = time.time()
        self.word_bag = [word for word in tokens if word not in self.stop_words]
        print "Time to compute word bag: ", (time.time() - start_time)/60., "minutes"

    def make_nltk_object_from_word_bag(self, word_bag=None):
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
        if word_bag is None:
            word_bag = self.word_bag
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
        """ Creates pandas dataframe called word_freq_df of the most common n
        words in corpus, with columns:

        occurrences: how often each of them occurred
        frequency: word frequency in the corpus
        frequency ratio: word relative frequency to background
        log frequency ratio: log of the relative frequency to background rates
        background_occur: the number of times word appears in background corpus

        (The log is useful because, for example, a rate two times as high as
        background has log ratio of +x, and a rate two times lower than
        background rates has a log ratio of -x.)

        n is the number of words we want to see. These words are draw in order
        of how frequently they are found in the corpus, so a large number of
        words should be chosen to make sure we find the interesting ones that
        appear much more often than in background corpus. (If a word appears
        often in our search corpus it may be because it also appear often in
        the background corpus, which is not of interest.)

        The actual words that were searched to collect the corpus are omitted
        from this dataframe (as long as self.search_terms has been set).

        n (int): number of most frequent words we want to appear in dataframe
        """
        start_time = time.time()
        # make dataframe we'll use in plotting
        num_words = n
        word_frequencies_list = []
        for word, occurrences in self.freq_dist.most_common(num_words):
            # determine whether word appears in background dict; if it does
            # not, the frequency ratio is set to zero
            if word in self.search_terms:
                continue
            if word in self.background_dict.keys():
                freq_ratio = self.freq_dist.freq(word)/self.background_dict[word][0]
                background_freq = self.background_dict[word][0]
                log_freq_ratio = log(freq_ratio)
                background_occur = self.background_dict[word][1]
            else:
                freq_ratio = 0
                background_freq = 0
                log_freq_ratio = 0
                background_occur = 0

            # faster to make list and then make dataframe in one line
            # than to repeatedly append to an existing dataframe
            word_frequencies_list.append((word, occurrences,
                                          self.freq_dist.freq(word),
                                          freq_ratio, log_freq_ratio,
                                          background_occur))
        word_freq_df = pd.DataFrame(word_frequencies_list,
                                columns=['word', 'occurrences', 'frequency',
                                'relative frequency', 'log relative frequency',
                                'background_occur'])
        print "Time to create word_freq_df: ", (time.time() - start_time)/60., "minutes"
        self.word_freq_df = word_freq_df

    def custom_word_frequency_dataframe(self, words):
        """ Same function as top_word_frequency_dataframe except instead of
        using top n words from corpus, a custom list of words is used. This
        function returns the dataframe it creates instead of setting it to
        word_freq_df. (The user can append what this function creates to
        word_freq_df by hand with pd.concat(df1, df1). )

        words: list of words to put in dataframe - each word is a string
        """

        word_frequencies_list = []
        words = [x.decode("utf-8") if type(x) == str else x for x in words]

        for word in words:
            # determine whether word appears in both background dict and corpus
            # if it does not, the frequency ratio is set to zero
            if word in self.search_terms:
                continue
            occurrences = self.freq_dist[word]
            if word in self.background_dict.keys() and occurrences != 0:
                freq_ratio = self.freq_dist.freq(word)/self.background_dict[word][0]
                background_freq = self.background_dict[word][0]
                log_freq_ratio = log(freq_ratio)
                background_occur = self.background_dict[word][1]
            else:
                freq_ratio = 0
                background_freq = 0
                log_freq_ratio = 0
                background_occur = 0

            # faster to make list and then make dataframe in one line
            # than to repeatedly append to an existing dataframe
            word_frequencies_list.append((word, occurrences,
                                          self.freq_dist.freq(word),
                                          freq_ratio, log_freq_ratio,
                                          background_occur))
        word_freq_df = pd.DataFrame(word_frequencies_list,
                                columns=['word', 'occurrences', 'frequency',
                                'relative frequency', 'log relative frequency',
                                'background_occur'])
        return word_freq_df

    def plot_word_frequencies(self, plot_string, dataframe=None):
        """ Plots of given value about word, where plot_string is a string
        that gives quantity to be plotted.

        Note that the plot can't display unicode characters correctly, so if a
        word looks like a little box you'll have to pull up word_freq_df to see
        what the character actually is.

        plot_string (string): column of word_freq_df dataframe, e.g.
                              "occurrences", "frequency", "relative frequency",
                              "log relative frequency", etc.
        dataframe (pandas dataframe): dataframe of the same form as
                                      word_freq_df; if left empty then
                                      self.word_freq_df is plotted
        """
        if dataframe is None:
            dataframe = self.word_freq_df

        num_words = len(dataframe)
        try:
            dataframe.set_index("word")[plot_string].plot.barh(figsize=(20,
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
        Term is a string.

        The returned object is a dataframe that contains the rows of tweets_df
        dataframe that have tweets containing term.

        term (string): term of interest
        """
        assert type(term) in (str, unicode)
        assert term

        pd.set_option('display.max_colwidth', -1)
        tweets_containing = self.tweets_df[self.tweets_df.text.str.contains(term) == True]
        print len(tweets_containing), "tweets contain this term"
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

        username (string): username of interest
        """
        assert type(username) in (str, unicode)
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
