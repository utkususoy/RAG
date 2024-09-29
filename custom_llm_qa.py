from click import prompt
from langchain.chains import RetrievalQA

from langchain.vectorstores import Chroma
# from langchain.retrievers import VectorStoreRetriever

# # Use the vector store retriever
# retriever = VectorStoreRetriever(vector_store)

# Define a template prompt
# template = """
# You are an AI assistant. You are given the following documents retrieved based on a query. Please answer the query in a conversational manner.
#
# Documents: {documents}
# Query: {query}
# """

# prompt = PromptTemplate(input_variables=["documents", "query"], template=template)


# Combine the retriever and the LLM to create a Retrieval-Augmented Generation (RAG) system
class CustomLlmQA(RetrievalQA):
    def __init__(self, llm, retriever=None):
        self.llm = llm
        self.retriever = retriever

    def run_rag(self, query: str, prompt):
        # Retrieve relevant documents
        #TODO: Check this
        docs = self.retriever.invoke(query)
        doc_texts = "\n".join([doc.page_content for doc in docs])

        # Generate a response
        full_prompt = prompt.format(documents=doc_texts, query=query)
        response = self.llm.generate(full_prompt)
        return response

    def run(self, query: str, prompt):
        full_prompt = prompt.format(query=query)
        response = self.llm.generate(full_prompt)
        return response

# # Use the retriever and the LLM
# rag_system = CustomRetrievalQA(llm, retriever, prompt)
#
# # Example query
# response = rag_system.run("What is Hugging Face?")
# print(response)
