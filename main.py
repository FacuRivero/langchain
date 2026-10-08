import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

load_dotenv()
print("Initializing components..")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

vectorstore = PineconeVectorStore(
  index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

# Retrieve a vector_store with search capabilities 
retriever = vectorstore.as_retriever(search_kwargs={"k":3}) # 3 Top documents

prompt_template = ChatPromptTemplate.from_template(
  """
Answer the question based only on the following context:
{context}

Question: {question}
Provide a detailed answer: 
"""
)

#
# IMPLEMENTATION 1: Without LCEL (Langchain Expresion language)
#
def format_docs(docs):
  """Format Retrieved documents into a single string."""
  return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_tool(query: str):
  """
  Simple retrieval chain without LCEL.
  Limitations:
  - Manual steb-by-step execution
  - No built-in stremaing support
  - No async support without additional code
  - Harder to compose with other chains
  - More verbose and error prone
  """
  # step 1: Retrieve relevant documents
  docs = retriever.invoke(query)
  # step 2: Format documents into context string
  context = format_docs(docs)
  # step 3: Format the prompt with the template and the query 
  messages = prompt_template.format_messages(context=context, question=query)
  # step 4: Invoke LLM with the formatted messages 
  response = llm.invoke(messages)
  return response.content


#
# IMPLEMENTATION 2: With LCEL (Langchain Expresion language) - BETTER APPROACH
#
def create_retrieval_chain_with_lcel():
  """
  Create a retrieval chain using LCEL.
  Returns a chain that can be invoked with ("questios": "...")

  Adventages over non-LCEL approach:
  - Declarative and composable: easy to chain operations with pipe operator (|)
  - Built-in streaming: chain.stream() works out of the box
  - Built-in async: chain.ainvoke() and chain.astream() available
  - Barch processing: chain.batch() for multiple inputs
  - Type safety: Better integration with Langchain's type system
  - less code: mORE CONSICE AND READABLE
  - Reusable: chain can be saved, shared and composed with other chains
  - Better debugging: Langchain provides better observability tools
  """

 # Result of the RunnablePAsstrough: COMBINES ORIGINAL + NEW
 #    Original = {"question": "What is Pinecone?"}
 #    New = {"context": "doc1\ndoc2\ndoc3" }
 # SO => { "question": "What is Pinecone?", "context": "doc1\ndoc2\ndoc3" }

 # itemgetter("question") extrae el valor de "question" del diccionario: "What is pinecone?"
 # retriever usa esa pregunta para buscar documentos relevantes: [doc1, doc2, doc3]
 # format_docs convierte esos documentos en un texto: "doc1\ndoc2\ndoc3"
 # assign(context=...) agrega ese resultado al diccionario original{ "question": "What is Pinecone?", "context": "doc1\ndoc2\ndoc3" }

  retrieval_chain = (
    RunnablePassthrough.assign(
      context=itemgetter("question") | retriever | format_docs
    )
    | prompt_template
    | llm
    | StrOutputParser()
  )
  return retrieval_chain

if __name__ == "__main__":
  print("Retrieving")

  query = "What is Pinecone in machine learning?"

  # Option 0: Raw invocation without RAG
  print("\n" + "=" * 70)
  print("IMPLEMENTATION 0: Raw LLM Invocation (NO RAG)")
  print("=" * 70)
  result_raw = llm.invoke([HumanMessage(content=query)])
  print("\nAnswer: ")
  print(result_raw.content)

  # Option 1: Use implementation without LCEL
  print("\n" + "=" * 70)
  print("IMPLEMENTATION 1: Without LCEL")
  print("=" * 70)
  result_without_lcel = retrieval_chain_without_tool(query)
  print("\nAnswer: ")
  print(result_without_lcel)


  # Option 2: Use implementation with LCEL
  print("\n" + "=" * 70)
  print("IMPLEMENTATION 1: With LCEL")
  print("=" * 70)
  chain_with_lcel = create_retrieval_chain_with_lcel()
  result_with_lcel = chain_with_lcel.invoke({"question": query})
  print(result_with_lcel)
  