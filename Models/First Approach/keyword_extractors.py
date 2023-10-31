################ Keywords #########################
from rake_nltk import Rake as rake
from rake_new2 import Rake as rake_new2
import nltk
import string
import networkx as nx
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from yake import KeywordExtractor
# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# RAKE
 ## First, we will extract the keywords
def extract_keywords_rake(text, num_keywords=5):
    ''' This function exrtacts the keywords using rake library '''
    r = rake()
    r.extract_keywords_from_text(text)
    keywords = r.get_ranked_phrases()[:num_keywords]
    return keywords
#RAKE_NEW2
def extract_keywords_rake_new2(text, num_keywords=5):
    ''' This function extracts the keywords using rake_new2 '''
    r = rake_new2()
    r.get_keywords_from_raw_text(text)
    keywords = list(r.get_ranked_keywords())[:num_keywords]
    return keywords
# TEXTRANK
def extract_keywords_textrank(text, num_keywords=5):
    '''  This function extracts the keywords using textrank '''
    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence) for sentence in sentences]

    # Remove punctuation and convert to lowercase
    words = [[word.lower() for word in sentence if word not in string.punctuation] for sentence in words]

    # Filter out stopwords
    stop_words = set(stopwords.words('english'))
    words = [[word for word in sentence if word not in stop_words] for sentence in words]

    # Build a graph based on cosine similarity of word vectors
    graph = nx.Graph()
    for sentence in words:
        for word in sentence:
            if word not in graph.nodes():
                graph.add_node(word)
        for word1 in sentence:
            for word2 in sentence:
                if word1 != word2:
                    if not graph.has_edge(word1, word2):
                        graph.add_edge(word1, word2, weight=1.0)

    # Calculate PageRank scores for nodes (words)
    scores = nx.pagerank(graph, weight='weight')

    # Get the top N keywords based on PageRank scores
    sorted_keywords = sorted(scores, key=scores.get, reverse=True)[:num_keywords]

    return sorted_keywords

#YAKE
def extract_keywords_yake(text, num_keywords=5):
    '''  This function extracts the keywords using yake  '''
    keyword_extractor = KeywordExtractor()
    keywords = keyword_extractor.extract_keywords(text)
    top_keywords = [keyword[0] for keyword in keywords][:num_keywords]
    return top_keywords