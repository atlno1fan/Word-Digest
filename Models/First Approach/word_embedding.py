#################### Word embedding #######################
from sentence_transformers import SentenceTransformer
import openai
import os
import numpy as np
from InstructorEmbedding import INSTRUCTOR
## Now, the word embeddings
# DAVINCI-001

def create_embedding_davinci( text):
      ''' This function will calculate the embedding using davinci model '''
      response = openai.Embedding.create(
          input=text,
          engine="text-similarity-davinci-001",
          gpu=True  # Use GPU for the operation
      )
      return response
#ADA-002
def create_embedding_ada( text):
      ''' This function will calculate the embedding using the ada model '''
      openai.api_key = "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"


      response = openai.Embedding.create(
          input=text,
          model="text-embedding-ada-002",
          gpu=True  # Use GPU for the operation
      )
      return response

