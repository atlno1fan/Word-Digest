#################### Sentence embedding ##############
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from sentence_transformers import SentenceTransformer, util
import torch

## We will continue with the sentence embedding
#Doc2vec
def generate_emb_doc_2_vec(text):
    ''' This function embeds the sentences with Doc2Vec model '''
    text=[text]
    # Tokenize the input text
    tokenized_text = [word_tokenize(sentence.lower()) for sentence in text]

    # Create TaggedDocument objects
    tagged_data = [TaggedDocument(words=words, tags=[str(i)]) for i, words in enumerate(tokenized_text)]

    # Initialize and train the Doc2Vec model
    doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=20)
    doc2vec_model.build_vocab(tagged_data)
    doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

    # Infer embeddings for each paragraph
    doc2vec_paragraph_embeddings = [doc2vec_model.infer_vector(words) for words in tokenized_text]
    doc2vec_paragraph_embeddings_1= doc2vec_paragraph_embeddings[0]
    doc2vec_paragraph_embeddings_2 = torch.tensor(doc2vec_paragraph_embeddings_1) # Convert to tensor
    return doc2vec_paragraph_embeddings_1,doc2vec_paragraph_embeddings_2


# SBERT
def generate_emb_sbert(text):
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        embeddings_1 = model.encode([text], convert_to_tensor=False)[0]  
        embeddings_2 = model.encode([text], convert_to_tensor=True)# Convert to tensor
        return embeddings_1,embeddings_2


