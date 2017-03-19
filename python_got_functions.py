""" Old funtions that called the python version of GetOldTweets -
not currently used in Twords, though could be added back in.

Experiments found that that python version of GetOldTweets was more prone
to hangups for collecting large numbers of tweets than the Java version.
The python version has since been updates though, so interested parties can
take a look at reintegrating it with Twords.
"""

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
