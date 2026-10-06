import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
load_dotenv()

def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Falta OPENAI_API_KEY en el entorno o en langchain-course/.env")

    print("Hello from langchain-course!")
    information = """
   Elon Reeve Musk (/ˈiːlɒn mʌsk/ Pretoria, 28 de junio de 1971) es un empresario, inversor, activista político conservador[3][4] y magnate sudafricano naturalizado estadounidense.[nota 1] Es el fundador, consejero delegado e «ingeniero» en jefe de la empresa SpaceX; inversor ángel, director general y arquitecto de productos de Tesla, Inc.; fundador de The Boring Company; y cofundador de Neuralink y OpenAI.[nota 2] Además, es el director de tecnología de X Corp.[5] Entre enero y mayo de 2025, ejerció como administrador de facto del Departamento de Eficiencia Gubernamental de la Casa Blanca bajo la segunda presidencia de Donald Trump.[6][7]A junio de 2026, Musk es la persona más rica del mundo según Forbes. Tras la salida a bolsa de SpaceX, su patrimonio neto superó los 1,1 billones de dólares estadounidenses, convirtiéndolo en el único y primer billlonario en términos de dólares estadounidenses de la historia.[8] El 16 de junio de 2026, su patrimonio alcanzaría un récord máximo de 1,4 billones de dólares.[9
   """

    summary_template = """
     given the information {information} about a person i WANT YOU TO CREATE:
     1. A SHORT SUMMARY
     2. TWO INTERESTING FACTS ABOUT THEM
   """

    summary_prompt_template = PromptTemplate(
      input_variables=["information"], template=summary_template
    )

    # llm = ChatOpenAI(temperature=0, model="gpt-5")
    llm = ChatOllama(temperature=0, model="gemma3:270m")
    chain = summary_prompt_template | llm
    # It inputs the output of the first step to the right component. 
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
