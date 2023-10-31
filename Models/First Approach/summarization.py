##################### Summary #######################
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
from collections import defaultdict
from rouge_score import rouge_scorer
## Now, the summarization libraries
#TEXTRANK
def textrank( text, max_tokens):
        ''' This function summarize the text using the textrank model '''
        stop_words = stopwords.words('english')
        words = [word for word in nltk.word_tokenize(text) if word not in stop_words]
        freq = defaultdict(int)
        for word in words:
            freq[word] += 1

        scores = defaultdict(int)
        summary_tokens = []
        token_count = 0

        for i, sentence in enumerate(nltk.sent_tokenize(text)):
            sentence_tokens = nltk.word_tokenize(sentence)
            if token_count + len(sentence_tokens) <= max_tokens:
                summary_tokens.extend(sentence_tokens)
                token_count += len(sentence_tokens) + 1  # Add 1 for space
            else:
                break

        summary = ' '.join(summary_tokens)
        return summary
# Lexrank
def lexrank( text, max_tokens):
        ''' This function summarize the text using the lexrank model '''
        stop_words = set(stopwords.words('english'))
        sentences = nltk.sent_tokenize(text)

        summary_tokens = []
        token_count = 0

        for i, sent1 in enumerate(sentences):
            if token_count >= max_tokens:
                break

            s1 = set(nltk.word_tokenize(sent1)) - stop_words
            sentence_tokens = nltk.word_tokenize(sent1)

            if token_count + len(sentence_tokens) <= max_tokens:
                summary_tokens.extend(sentence_tokens)
                token_count += len(sentence_tokens) + 1  # Add 1 for space
            else:
                break

        summary = ' '.join(summary_tokens)
        return summary
#LUHN
def luhn( text, max_tokens):
        ''' This function summarize the text using the luhn model '''
        stop_words = stopwords.words('english')
        sentences = nltk.sent_tokenize(text)

        freq = defaultdict(int)
        for sentence in sentences:
            for word in nltk.word_tokenize(sentence):
                if word not in stop_words:
                    freq[word] += 1

        summary_tokens = []
        token_count = 0

        for i, sentence in enumerate(sentences):
            if token_count >= max_tokens:
                break

            sentence_tokens = nltk.word_tokenize(sentence)

            if token_count + len(sentence_tokens) <= max_tokens:
                summary_tokens.extend(sentence_tokens)
                token_count += len(sentence_tokens) + 1  # Add 1 for space
            else:
                break

        summary = ' '.join(summary_tokens)
        return summary