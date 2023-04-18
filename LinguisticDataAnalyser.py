import statistics
import textstat
from scipy.stats import ttest_ind
import pandas as pd
from nltk.tokenize import *
import nltk.data
from nltk import FreqDist, RegexpParser
from nltk.corpus import swadesh
from nltk.corpus import stopwords
import numpy as np
import regex as re
import string
from nltk.tree import Tree


def test():
    debates_df = pd.read_csv('TD_TSV', sep='\t')
    twitter_df = pd.read_csv('MaryLouMcDonald-Twitter.csv', sep='\t', header=None)
    DataAnalyserML = DataAnalyser(twitter_df, debates_df)


class DataAnalyser:

    def __init__(self, twitter_df, debates_df):

        self.twitter_df = twitter_df
        self.debates_df = debates_df
        self.statements_df = None
        self.questions_df = None
        self.padStatementsAndQuestions()
        self.liguisitic_complexities()

    def padStatementsAndQuestions(self):
        t_df = self.twitter_df
        d_df = self.debates_df
        df_splits = [v for k, v in d_df.groupby('forum')]
        self.statements_df = df_splits[0]
        print(len(self.statements_df))
        self.questions_df = df_splits[1]
        print(len(self.questions_df))
        print("debate sections into respected dataframes")

    def liguisitic_complexities(self):

        questions_text = ' '.join(self.questions_df["text"])
        statements_text = ' '.join(self.statements_df["text"])
        twitter_text = ' '.join(self.twitter_df[9])

        forum_stats = {'social_media': None, 'statements': None, 'questions': None}
        for key in forum_stats.keys():
            forum_stats[key] = {'readability': 0, 'avg_wl': 0, 'ttr': 0, 'lw_distrib': 0,'sw_distrib': 0
                ,'avg_tree_comp': 0, 'sent_mean': 0,'sent_mode': 0, 'sent_median': 0, 'sent_variance':0}

        smr = self.readability_Flesch_Kincaid(twitter_text)
        sr = self.readability_Flesch_Kincaid(statements_text)
        qr = self.readability_Flesch_Kincaid(questions_text)

        print("Initial Social Media Readability grade level:", smr)
        forum_stats['social_media']['readability'] = smr
        print("Initial Statements Readability grade level:", sr)
        forum_stats['statements']['readability'] = sr
        print("Initial Questions Readability grade level:", qr)
        forum_stats['questions']['readability'] = qr
        print("\n")

        twitter_text = self.remove_urls_and_handles(twitter_text)

        print("texts setup")

        forum_texts_dic = {'questions': questions_text, 'statements': statements_text, 'social_media': twitter_text}
        for forum in forum_texts_dic:
            print("for the forum ", forum)

            avg_w = self.average_word_length(forum_texts_dic[forum])
            print("The average word length is       :", avg_w)
            forum_stats[forum]['avg_wl'] = avg_w

            ttr = self.ttr(forum_texts_dic[forum])
            print("The Type-Token Ratio is          :", ttr)
            forum_stats[forum]['ttr'] = ttr

            lw_distribut = self.large_word_distribution(forum_texts_dic[forum])
            print("The large_word distribution is   :", lw_distribut)
            forum_stats[forum]['lw_distrib'] = lw_distribut

            sw_distribut = self.short_word_distribution(forum_texts_dic[forum])
            print("The short_word distribution is   :", sw_distribut)
            forum_stats[forum]['sw_distrib'] = sw_distribut

            avg_tree_complexity = self.average_tree_complexity(forum_texts_dic[forum])
            print("The average tree complexity is   :", avg_tree_complexity)
            forum_stats[forum]['avg_tree_comp'] = avg_tree_complexity
            print("\n")


        dail_forums = ['statements', 'questions']

        for forum in dail_forums:
            sent_stats = self.sentence_length_stats_dail(forum_texts_dic[forum])
            mean,median,mode, variance = sent_stats[0], sent_stats[1], sent_stats[2], sent_stats[3]
            forum_stats[forum]['sent_mean'] = mean
            forum_stats[forum]['sent_mode'] = mode
            forum_stats[forum]['sent_median'] = median
            forum_stats[forum]['sent_variance'] = variance
            print("Sentence stats for forum ", forum,"\nshow mean, mode and median length of: ", mean, ", ", mode, ", ", median)
            print("and variance of : ", variance, "\n")

        twitter_sent_stats = self.sentence_length_stats_twitter()
        mean_t, median_t, mode_t, variance_t = twitter_sent_stats[0], twitter_sent_stats[1], twitter_sent_stats[2], twitter_sent_stats[3]
        forum_stats['social_media']['sent_mean'] = mean_t
        forum_stats['social_media']['sent_mode'] = mode_t
        forum_stats['social_media']['sent_median'] = median_t
        forum_stats['social_media']['sent_variance'] = variance_t
        print("Sentence stats for forum social media", "\nshow mean, mode and median length of: ", mean_t, ", ", median_t, ", ",
              mode_t)
        print("and variance of : ", variance_t, "\n")
        print("Complexity Analysed")


        print("Signifigance Testing")

        s_sent_count = self.sentence_count(forum_texts_dic['statements'])
        q_sent_count = self.sentence_count(forum_texts_dic['questions'])
        t_sent_count = self.sentence_count_twitter()

        self.text_complexity_significance((forum_stats['statements']['sent_mean']), (forum_stats['questions']['sent_mean']),
                                          s_sent_count,q_sent_count, 0.05 )
        self.text_complexity_significance((forum_stats['questions']['sent_mean']),
                                          (forum_stats['statements']['sent_mean']),
                                          q_sent_count, s_sent_count, 0.05)
        self.text_complexity_significance((forum_stats['social_media']['sent_mean']),
                                          (forum_stats['statements']['sent_mean']),
                                          t_sent_count, s_sent_count, 0.05)
        self.text_complexity_significance((forum_stats['statements']['sent_mean']),
                                          (forum_stats['social_media']['sent_mean']),
                                          t_sent_count, s_sent_count, 0.05)
        self.text_complexity_significance((forum_stats['questions']['sent_mean']),
                                          (forum_stats['social_media']['sent_mean']),
                                          q_sent_count, t_sent_count, 0.05)
        self.text_complexity_significance((forum_stats['social_media']['sent_mean']),
                                          (forum_stats['questions']['sent_mean']),
                                          t_sent_count, q_sent_count, 0.05)

        print("\n**********************************************************\n")
        s_w_count = self.word_count(forum_texts_dic['statements'])
        q_w_count = self.word_count(forum_texts_dic['questions'])
        t_w_count = self.word_count_twitter()
        self.text_complexity_significance((forum_stats['statements']['avg_wl']), (forum_stats['questions']['avg_wl']),
                                          s_w_count,q_w_count, 0.05)
        self.text_complexity_significance((forum_stats['questions']['avg_wl']),
                                          (forum_stats['statements']['avg_wl']),
                                          q_w_count, s_w_count, 0.05)
        self.text_complexity_significance((forum_stats['social_media']['avg_wl']),
                                          (forum_stats['statements']['avg_wl']),
                                          t_w_count, s_w_count, 0.05)
        self.text_complexity_significance((forum_stats['statements']['avg_wl']),
                                          (forum_stats['social_media']['avg_wl']),
                                          s_w_count, t_w_count, 0.05)
        self.text_complexity_significance((forum_stats['questions']['avg_wl']),
                                          (forum_stats['social_media']['avg_wl']),
                                          q_w_count, t_w_count, 0.05)
        self.text_complexity_significance((forum_stats['social_media']['avg_wl']),
                                          (forum_stats['questions']['avg_wl']),
                                          t_w_count, q_w_count, 0.05)


    def sentence_length_stats_dail(self, text):#return mean median and mode of sent length for dail entries
        sent_list = sent_tokenize(text)
        sentence_lengths = []

        number=0
        for sentence in sent_list:
            words = sentence.split()
            sentence_lengths.append(len(words))
            number = number+1

        print(number)

        mean_length = statistics.mean(sentence_lengths)
        median_length = statistics.median(sentence_lengths)
        mode_length = statistics.mode(sentence_lengths)
        variance_length = statistics.variance(sentence_lengths)
        sent_stats = (mean_length, mode_length, median_length, variance_length)
        return sent_stats

    def sentence_length_stats_twitter(self):#return mean median and mode of sent length for twitter entries
        twitter_sentence_lengths = []

        for tweet in self.twitter_df[9]:
            words = tweet.split()
            twitter_sentence_lengths.append(len(words))

        mean_length = statistics.mean(twitter_sentence_lengths)
        median_length = statistics.median(twitter_sentence_lengths)
        mode_length = statistics.mode(twitter_sentence_lengths)
        variance_length = statistics.variance(twitter_sentence_lengths)
        sent_stats = (mean_length, mode_length, median_length, variance_length)
        return sent_stats

    # Average word length measure word complexity in text
    def average_word_length(self, text):

        words = self.lower_words_from_text(text)
        total_chars = sum(len(word) for word in words)
        avg_length = total_chars / len(words)
        return avg_length

    # Type-Token Ratio diversity of vocabulary in text
    def ttr(self, text):
        lc_text_words = self.lower_words_from_text(text)
        unique_words = set(lc_text_words)
        return len(unique_words) / len(lc_text_words)

    def average_tree_complexity(self, text):

        sentences = sent_tokenize(text)
        tree_complexities = []
        for sentence in sentences:
            tree_complexities.append(self.tree_complexity(sentence))
        return sum(tree_complexities) / len(tree_complexities)

    # PST complexity given a sentence
    def tree_complexity(self, sent):
        tokens = nltk.word_tokenize(sent)
        tokens = [word.lower() for word in tokens]
        tokens_tagged = nltk.pos_tag(tokens)
        chunker = RegexpParser("""
                       NP: {<DT>?<JJ>*<NN>}    #To extract Noun Phrases
                       P: {<IN>}               #To extract Prepositions
                       V: {<V.*>}              #To extract Verbs
                       PP: {<p> <NP>}          #To extract Prepositional Phrases
                       VP: {<V> <NP|PP>*}      #To extract Verb Phrases
                       """)
        parse_output = chunker.parse(tokens_tagged)
        height = parse_output.height()
        width = self.tree_width(parse_output)
        tree_complexity = height * width
        return tree_complexity

    def large_word_distribution(self, text):
        large_words = 0
        words = self.lower_words_from_text(text)
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w.lower() in stop_words]
        for word in words:
            if len(word) > 6:
                large_words = large_words + 1
        large_word_distribtution = large_words / len(words)
        return large_word_distribtution

    def short_word_distribution(self, text):
        short_words = 0
        words = self.lower_words_from_text(text)
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w.lower() in stop_words]
        for word in words:
            if len(word) <= 4 :
                short_words = short_words + 1
        short_word_distribtution = short_words / len(words)
        return short_word_distribtution

    def readability_Flesch_Kincaid(self, text):
        # Calculate the Flesch-Kincaid grade level
        flesch_kincaid_grade_level = textstat.flesch_kincaid_grade(text)
        # based on 0.39 * (total words / total sentences) + 11.8 * (total syllables / total words) - 15.59
        return flesch_kincaid_grade_level

    def lower_words_from_text(self, text):
        words = text.lower().split()
        return words

    def remove_urls_and_handles(self, text):
        text = self.strip_links(text)
        text = self.strip_all_handles(text)
        return text

    def strip_links(self, text):
        link_regex = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
        links = re.findall(link_regex, text)
        for link in links:
            text = text.replace(link[0], ', ')
        return text

    def strip_all_handles(self, text):
        entity_prefixes = ['@', '#']
        for separator in string.punctuation:
            if separator not in entity_prefixes:
                text = text.replace(separator, ' ')
        words = []
        for word in text.split():
            word = word.strip()
            if word:
                if word[0] not in entity_prefixes:
                    words.append(word)
        return ' '.join(words)

    def tree_width(self, tree):
        max_width = 0
        for pos in tree.treepositions():
            if len(pos) > max_width:
                max_width = len(pos)
        return max_width


    def text_complexity_significance(self,text1_mean_len, text2_mean_len, text1_count, text2_count, alpha):
        """
        Perform a significance test on text complexity using the mean sentence length as a proxy.

        Parameters:
        text1_mean_len (float): The average sentence length of text 1.
        text2_mean_len (float): The average sentence length of text 2.
        text1_count (int): The number of sentences in text 1.
        text2_count (int): The number of sentences in text 2.
        alpha (float): The level of significance for the test.

        Returns:
        A tuple containing the t-value and the p-value of the test.
        """

        # Calculate the pooled standard deviation
        pooled_std_dev = ((text1_count - 1) * (text1_mean_len ** 2) + (text2_count - 1) * (text2_mean_len ** 2)) / (
                    text1_count + text2_count - 2)
        pooled_std_dev = pooled_std_dev ** 0.5

        # Calculate the t-value
        t_value = (text1_mean_len - text2_mean_len) / (pooled_std_dev * ((1 / text1_count) + (1 / text2_count)) ** 0.5)

        # Calculate the p-value
        p_value = ttest_ind(list(range(text1_count)), list(range(text2_count)), equal_var=False).pvalue

        # Check if the p-value is less than the level of significance
        if p_value < alpha:
            print(f"The test is significant at alpha = {alpha} with t-value = {t_value} and p-value = {p_value}")
        else:
            print(f"The test is not significant at alpha = {alpha} with t-value = {t_value} and p-value = {p_value}")

        return t_value, p_value

    def sentence_count(self,text):
        sentences = sent_tokenize(text)
        return len(sentences)

    def sentence_count_twitter(self):
        tweet_count = 0
        for sentence in self.twitter_df[9]:
            tweet_count = tweet_count+1
        return  tweet_count


    def word_count(self, text):
        words = self.lower_words_from_text(text)
        return len(words)

    def word_count_twitter(self):
        tweet_count = 0
        for sentence in self.twitter_df[9]:
            words = word_tokenize(sentence)
        return  len(words)


# def signifigance_testing(self):

if __name__ == '__main__':
    test()
