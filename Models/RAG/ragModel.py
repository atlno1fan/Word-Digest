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

    def get_paragraphs_RAG(self, text):
        self.question, link = self.check_link_in_text(text)
        if self.link != link:
            self.paragraphs, self.titles = pdf_to_html(link)

            (
                self.chunked_corpus,
                self.passage_titles,
                self.passages,
            ) = paragraph_splitter(self.paragraphs, self.titles)

            self.input_ids = tokenize_paragraph(self.chunked_corpus)

            self.ctx_encoder, self.device = set_device()

            self.embeddings = generate_embeddings(
                self.input_ids, self.ctx_encoder, self.device
            )

            self.index, self.dim, self.m = add_faiss_index(self.embeddings)

            self.dataset = generate_dataset(
                self.chunked_corpus,
                self.embeddings,
                self.index,
                self.dim,
                self.m,
                self.passage_titles,
                self.passages,
            )

            self.tokenizer, self.model = self.intialize_model(self.dataset)
            self.link = link
            # print(link)
        # else:
        #    print('link is the same')

        input_ids = self.tokenizer(self.question, return_tensors="pt").input_ids.to(
            self.device
        )

        # Get question embeddings
        question_embeddings = self.model.question_encoder(input_ids)[0]

        # Convert to NumPy
        question_embeddings = question_embeddings.detach().cpu().numpy()

        docs = self.model.retriever.retrieve(question_embeddings, n_docs=5)

        generated = self.model.generate(input_ids=input_ids)

        encoded, doc_ids, doc_texts = docs
        target_paragraph = "\n".join(doc_texts[0]["text"])
        return target_paragraph
    
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
    
    
