from transformers import AutoTokenizer, AutoModel
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
import fitz  # PyMuPDF
import torch
from langchain_chroma.vectorstores import Chroma # https://python.langchain.com/v0.2/api_reference/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma

from typing import List

class CustomEmbedding(Embeddings):
    def __init__(self, model_name: str, device: str):
        #TODO: set Device config.
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_documents(self, texts: list) -> List[List[float]]:
        """Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """

        embeddings = []
        for text in texts:
            # inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
            # with torch.no_grad():
            #     outputs = self.model(**inputs)
            # embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
            embedding = self.embed_query(text)
            embeddings.append(embedding)# Mean pooling
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        # Tokenize the input text and move to the appropriate device
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)

        # Disable gradient calculation
        with torch.no_grad():
            # Get model outputs
            outputs = self.model(**inputs)

        # Apply mean pooling to get the average embedding (1D vector for the query)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # Move back to CPU for further processing

        return embedding


def extract_texts_from_pdf(pdf_file):
    """Extract texts from a PDF, split into chunks, and return LangChain Document objects."""
    doc = fitz.open(pdf_file)
    documents = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Define chunk size for splitting the text
        chunk_overlap=100  # Define overlap between chunks for better context continuity
    )

    # Iterate through all the pages
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text("text")

        # Use RecursiveCharacterTextSplitter to split the text into chunks
        chunks = splitter.create_documents([text], metadatas=[{"page": page_num + 1}])
        documents.extend(chunks)  # Add all the chunks for this page to the list

    return documents



if __name__ == "__main__":
    # Initialize the custom embedding model
    embedding_model = CustomEmbedding(model_name="sentence-transformers/all-mpnet-base-v2",
                                      device="cpu")

    # Store documents and their embeddings in the vector store (Chroma in this case)
    pdf_file = "attention.pdf"

    texts_ = extract_texts_from_pdf(pdf_file=pdf_file)

    vector_store = Chroma(
        collection_name="attention_collection",
        embedding_function=embedding_model,
        persist_directory="./attention_chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )
    vector_store.add_documents(documents=texts_)
