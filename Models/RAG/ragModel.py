from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import re
import requests
from pdfPreprocessingRAG import *


class RAG_target_paragraph_generator:
    def __init__(self, link, question):
        self.link = link
        self.question = question

    def intialize_model(self, dataset):
        tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
        # retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", use_dummy_dataset=False, indexed_dataset=dataset, index_name="embeddings")
        retriever = RagRetriever.from_pretrained(
            "facebook/rag-sequence-nq",
            use_dummy_dataset=False,
            indexed_dataset=dataset,  # pass your dataset here
            index_name="embeddings",
        )
        model = RagSequenceForGeneration.from_pretrained(
            "facebook/rag-sequence-nq", retriever=retriever
        ).to(self.device)

        return tokenizer, model


    def check_link_in_text(self, text):
        regex = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        links = re.findall(regex, text)

        for link in links:
            try:
                response = requests.head(link)
                if response.status_code == 200:
                    text = text.replace(link, "")
                    return text, link
            except:
                pass

        return text, self.link

    def prompt_generator(self, text):
        target_paragraph = self.get_paragraphs_RAG(text)
        prompt = (
            "Can you answer this question based on this text within the triple quotation markes? The question is"
            + self.question
            + " ''' "
            + target_paragraph
            + " '''"
        )
        return prompt
