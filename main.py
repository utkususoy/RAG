from rag_llm import RagLlm

if __name__ == '__main__':
    query = "What is an input sequence of symbol representations of Encoder Maps?"
    rag_llm_runner = RagLlm()
    res = rag_llm_runner.retrieval.invoke(query)
    print(res)