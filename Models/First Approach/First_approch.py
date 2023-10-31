import requests
from bs4 import BeautifulSoup
import numpy as np
import torch
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize, word_tokenize
import openai
from sentence_transformers import util
import torch
from sentence_transformers import SentenceTransformer
from keyword_extractors import extract_keywords_rake,extract_keywords_rake_new2,extract_keywords_textrank,extract_keywords_yake
from summarization import textrank,lexrank,luhn
from word_embedding import create_embedding_davinci,create_embedding_ada
from Sentence_embedding  import generate_emb_doc_2_vec,generate_emb_sbert
class First_approch:
  def __init__(self, url):
        self.url = url
        self.pdf_info=[]
        self.keyword_embeddings_per_paragraph={}
  
  
    
  # convert pdf to html then extract the paragraphes from the html 
  def to_paragraphs(self):
    ''' Ths function will take the pdf and convert it to HTML and extract paragraphs then return it '''
    api_url = "https://debi-api.azurewebsites.net/Words_Digest/GetHtmlFromUrl"
    # pdf_url = "https://www.uou.ac.in/sites/default/files/slm/Introduction-cyber-security.pdf"

    # bearer token from Postman
    BEARER_TOKEN = "1a9901afbe3ed24491c700513799093b42"

    headers = {
      "Authorization": f"Bearer {BEARER_TOKEN}"
    }

    params = {
      "url": self.url,
      "responseAsFileUrl": "false"
    }

    response = requests.get(api_url, headers=headers, params=params)
    html = response.text
    # html
    soup = BeautifulSoup(html, 'html.parser')

    paragraphs = []
    for p in soup.find_all('p'):
        paragraphs.append(p.text)

    return paragraphs
  
  def kw_emb_generator(self, paragraphs ,keyword_library, embeddings_library=generate_emb_doc_2_vec):
      ''' This function will extract the keyword  and embedding from the paragraphs of the pdf '''
      keyword_results = []
      embeddings_results = []

      for paragraph in paragraphs:
          keywords = keyword_library(paragraph)
          _,embeddings= embeddings_library(paragraph)

          keyword_results.append(keywords)
          embeddings_results.append(embeddings)

      return keyword_results, embeddings_results, paragraphs

  def kw_emb_store(self,keyword_library, embeddings_library,word_embedding_function):
      ''' This function will store keyword  and embedding in list of list in a variable called pdf_info '''
      paragraphs = self.to_paragraphs()
      keywords, embeddings, paragraphs = self.kw_emb_generator(paragraphs,keyword_library, embeddings_library)

      #pdf_info is the list with the pdf paragraphs and their keywords and embeddings

      pdf_info = [
      [paragraph_index,paragraphs ,keywords, embeddings]
      for paragraph_index, (paragraphs,keywords, embeddings) in enumerate(zip(paragraphs,keywords, embeddings))
        ]
      self.pdf_info = pdf_info
       # Create pairs of keywords and paragraph indices
      keyword_paragraph_pairs = []
      for sublist_index, sublist in enumerate(self.pdf_info):
          keyword_list = sublist[2]
          paragraph_index = sublist_index
          keyword_pairs = [(keyword, paragraph_index) for keyword in keyword_list]
          keyword_paragraph_pairs.extend(keyword_pairs)
      # print("keyword_paragraph_pairs",keyword_paragraph_pairs)
      # Calculate embeddings for keywords and associate them with paragraphs
      keyword_embeddings_per_paragraph = {}
      for keyword, paragraph_index in keyword_paragraph_pairs:
          embedding = word_embedding_function(keyword)['data'][0]['embedding']
          if paragraph_index not in keyword_embeddings_per_paragraph:
              keyword_embeddings_per_paragraph[paragraph_index] = []
          keyword_embeddings_per_paragraph[paragraph_index].append((embedding, keyword))
      
      self.keyword_embeddings_per_paragraph=keyword_embeddings_per_paragraph

      return pdf_info
   

  def keyword_compare(self,user_question,embedding_function,kw_function, similarity_threshold):
    ''' This function create pairs between the keyword and the indicies of the paragraph then calculate the embedding for the keyword and associate all of them with the paragraphs
    then calculate the similarity between the user keyword and the paragraph keyword
    finaly , paragraphs with these keywords that meets a certain thereshold'''

    # Prepare user input for keyword extraction and embedding
    user_info = [user_question]
    keywords_user, _, _ = self.kw_emb_generator(
        user_info, kw_function
    )
    flat_keywords_user = [keyword for keyword_list in keywords_user for keyword in keyword_list]
    embeddings_results_kw_em_user = [embedding_function(keyword)['data'][0]['embedding'] for keyword in flat_keywords_user]


    # Calculate average cosine similarity between user keywords and paragraph keywords
    average_similarities_per_paragraph = {}
    for paragraph_index, paragraph_embeddings in self.keyword_embeddings_per_paragraph.items():
        paragraph_similarities = []
        for user_embedding in embeddings_results_kw_em_user:
            similarities = []
            for keyword_embedding, _ in paragraph_embeddings:
                similarity = np.dot(user_embedding, keyword_embedding) / (np.linalg.norm(user_embedding) * np.linalg.norm(keyword_embedding))
                similarities.append(similarity)
            average_similarity = sum(similarities) / len(similarities) if len(similarities) > 0 else 0.0
            paragraph_similarities.append((average_similarity, paragraph_index))
        average_similarities_per_paragraph[paragraph_index] = paragraph_similarities

        # print(f"Average similarity for paragraph {paragraph_index}: {average_similarity}")

    # Find paragraphs with keywords that meet the similarity threshold
    similar_paragraphs = {}
    for paragraph_index, paragraph_similarities in average_similarities_per_paragraph.items():
        similar_indices = [idx for idx, (similarity, _) in enumerate(paragraph_similarities) if similarity >= similarity_threshold]
        if similar_indices:
            similar_keywords = [(idx, keyword) for idx, (_, keyword) in enumerate(self.keyword_embeddings_per_paragraph[paragraph_index]) if idx in similar_indices]
            similar_paragraphs[paragraph_index] = similar_keywords
            # print("similar_keywords",similar_keywords)

    # Extract the indices of similar paragraphs
    similar_paragraph_indices_ = list(similar_paragraphs.keys())
    # print(similar_indices)

    return similar_paragraph_indices_


  
  def embeddings_compare(self, paragraphs_indices, user_info, embedding_function):
        ''' this function takes the output of keyward compare function an d filter the top 5 paragraphes according to 
         similarity score betwee the paragraphes and the question embeddings '''
        # Get user input embedding
        _,user_input_embedding = embedding_function(user_info)
        # get the tensors of paragraph embeddings 
        embeddings_paragraphs = [self.pdf_info[idx][3] for idx in paragraphs_indices]
        #embeddings_paragraphs = [torch.tensor(embedding[0]) for embedding in embeddings_paragraphs]

        # Calculate cosine similarity between user input embedding and embeddings of similar paragraphs
        cosine_similarities = []
        for embedding in embeddings_paragraphs:
            if user_input_embedding.size(0) != embedding.size(0):  # Check if dimensions are compatible
                # Resize user input embedding to match paragraph embedding dimension
                resized_user_embedding = F.interpolate(user_input_embedding.unsqueeze(0), size=embedding.size(0), mode='linear', align_corners=False).squeeze(0)
                cosine_similarity = torch.mean(torch.cosine_similarity(resized_user_embedding, embedding, dim=0))
            else:
                cosine_similarity = torch.mean(torch.cosine_similarity(user_input_embedding, embedding, dim=0))
            cosine_similarities.append(cosine_similarity.item())

        # Create a list of tuples containing (paragraph_index, cosine_similarity)
        similarity_scores = list(zip(paragraphs_indices, cosine_similarities))

        # Sort the similarity scores in descending order based on cosine similarity
        sorted_similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Get the top 5 similar paragraph indices
        top_5_indices = [idx for idx, _ in sorted_similarity_scores[:5]]

        return top_5_indices
    



  def tokens_check_and_summerization(self,user_prompt, top_indices,summarization_library,token_threshold=500):
      ''' This function gets the combined paragraphs (from top 5 indecies) and get the size of the tokens
      then, do the same thing with the user prompt ,
      after that , we make the total token limit is equal to the subtraction of the the summary token limit(half the gpt prompt limit)(thresold) and user prompt token size
      and then, compare the size of the tokens of the combined paragraphs tototal token limit
      if the token size of the combined paragraphs is greater that the total token limit , then we will summarize the combained text
      otherwise, we will return the same text without summarization'''

      # Combine the selected paragraphs
      combined_paragraphs = [self.pdf_info[idx][1] for idx in top_indices]
      combined_text = ' '.join(combined_paragraphs)

      # Tokenize the combined text and calculate the token size
      tokens = word_tokenize(combined_text)
      token_size_paragraph = len(tokens)
      # print("token_size_paragraph",token_size_paragraph)

      # Tokenize the combined text and calculate the token size
      tokens = word_tokenize(user_prompt)
      token_size_user = len(tokens)
      # print("token_size_user ",token_size_user )


      # Define the summary tokens limit and calculate the total limit including the user's prompt token size
      summary_tokens_limit = token_threshold
      total_tokens_limit = summary_tokens_limit - token_size_user

      # Compare the token size to the specified threshold
      if token_size_paragraph > total_tokens_limit:

        # Use the specified summarization library to generate a summary
        summary= summarization_library(combined_text,total_tokens_limit)
        tokens_summary = word_tokenize(summary)
        token_size_summary = len(tokens_summary)
        # print("token_size_summary ",token_size_summary )
        ratio=(token_size_summary/token_size_paragraph)*100
        print("Ratio between token_size_summary and token_size_paragraph:",ratio)
      else:
        summary= combined_text
        print("Ratio between token_size_summary and token_size_paragraph: 100")
      return summary

  def prompt_generator(self,user_prompt,summary):
      ''' This function will get the user prompt and the summary and generate the prompt that will be passed to gpt '''
      prompt = user_prompt+'\n'+"answer only from the following paragraph even if the answer not provided say that "+'\n'+summary

      return prompt

  def run_chatgpt(self,conversation_history):
      ''' This function will use gpt-3.5-turbo to run gpt as it will answer the questions of the user from the pdf '''

      openai.api_key = "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"
      # Make API call
      api_response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=conversation_history
      )

      # Extract and print assistant's reply
      response = api_response.choices[0].message["content"]

      return response


  def evaluation(self, summarized_text, response, embedding_function):


      summarized_embedding = embedding_function(summarized_text)[0]
      response_embedding = embedding_function(response)[0]

      # Calculate cosine similarity
      cosine_similarity = util.pytorch_cos_sim([summarized_embedding], [response_embedding])[0][0]

      # Calculate and return similarity percentage
      similarity_percentage = (cosine_similarity + 1) / 2 * 100
      return similarity_percentage
 