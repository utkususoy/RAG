from custom_llm import CustomLLM
from custom_llm_qa import CustomLlmQA
from custom_embedding import CustomEmbedding
# from langchain.retrievers import VectorStoreRetriever
from langchain.prompts import PromptTemplate
from langchain_chroma.vectorstores import Chroma # https://python.langchain.com/v0.2/api_reference/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma

class RagLlm:
    def __init__(self):
        self.MODEL_NAME = ""
        self.LLM_MODEL_DEVICE = ""
        self.EMBEDDING_DEVICE = "cpu"
        self.EMBEDDING_NAME = "sentence-transformers/all-mpnet-base-v2"
        self.VECTOR_STORE_PATH = ""

        # self._configure_chat_prompt()
        # self._configure_rag_prompt()
        # self._load_llm_model()
        self._load_vector_store()

    def _load_llm_model(self):
        self.model = CustomLLM(model_name=self.MODEL_NAME, device=self.LLM_MODEL_DEVICE)

    def _load_embedding_model(self):
        return CustomEmbedding(model_name=self.EMBEDDING_NAME, device=self.EMBEDDING_DEVICE)

    def _load_vector_store(self):
        embedding_model = self._load_embedding_model()
        # TODO: Test CHromaVectorStore and Embedding
        vector_store = Chroma(
                            collection_name="attention_collection",
                            embedding_function=embedding_model,
                            persist_directory="./attention_chroma_langchain_db",  # Where to save data locally, remove if not necessary
                        )
        # TODO: Change Retrieval
        self.retrieval = vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 5},
                    )

    def _configure_rag_prompt(self):
        #TODO: Change Prompt
        rag_prompt = """
                    You are an AI assistant. You are given the following documents retrieved based on a query. Please answer the query in a conversational manner.

                    Documents: {documents}
                    Query: {query}
                    """
        self.rag_prompt_template = PromptTemplate(input_variables=["documents", "query"], template=rag_prompt)

    def _configure_chat_prompt(self):
        # TODO: Change Prompt
        self.chat_prompt = """
                    You are an AI assistant. Please answer the query in a conversational manner.

                    Query: {query}
                    """
        self.chat_prompt_template = PromptTemplate(input_variables=["query"], template=self.chat_prompt)

    def query_llm(self, query):
        response = self.model.generate(query)
        query_retrieval = CustomLlmQA(llm=self.model)
        query_retrieval.run(query=query,
                            prompt=self.chat_prompt_template)
        print(response)

    def query_rag_llm(self, query):
        rag_retrieval = CustomLlmQA(llm=self.model,
                                    retriever=self.retrieval)
        rag_retrieval.run_rag(query=query,
                              prompt=self.rag_prompt_template)




