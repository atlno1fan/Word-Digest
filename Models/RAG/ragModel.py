from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import re
import requests
from pdfPreprocessingRAG import *


class RAG_target_paragraph_generator:
    def init(self, link, question):
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