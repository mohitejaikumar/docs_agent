import asyncio
from operator import itemgetter

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from crawlers import crawl_sitemap

load_dotenv()

# app = FastAPI()


# @app.post("/crawl/")
# async def crawl_docs(url: str) -> None:
#     await crawl_sitemap(sitemap_url=url)

asyncio.run(crawl_sitemap(sitemap_url="https://python.langchain.com/sitemap.xml"))

# thread_id = 1

# memory = MemorySaver()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
)

template1 = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output (under 50 queries):"""

prompt_decomposition = ChatPromptTemplate.from_messages(
    [
        ("system", template1),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

# Chain
generate_queries_decomposition = (
    prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n"))
)

# Prompt
template2 = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""

decomposition_prompt = ChatPromptTemplate.from_template(template2)


def format_qa_pair(question, answer):
    """Format Q and A pair"""

    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()


# Chat history
chat_history = []


async def main():
    text_loader_kwargs = {"autodetect_encoding": True}
    loader = DirectoryLoader(
        "./output/",
        glob="**/*.txt",
        show_progress=True,
        use_multithreading=True,
        loader_cls=TextLoader,
        silent_errors=True,
        loader_kwargs=text_loader_kwargs,
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
    )
    retriever = vectorstore.as_retriever()

    # ### Build retriever tool ###
    # tool = create_retriever_tool(
    #     retriever,
    #     "docs_retriever",
    #     """
    # This tool is used to retrieve relevant documentation and code snippets for particular library and framework use to solve subproblem .
    # """,
    # )

    # tools = [tool]

    # agent_executor = create_react_agent(
    #     llm,
    #     tools,
    #     checkpointer=memory,
    # )

    while True:
        user_input = input("Question: ")
        if user_input.lower() == "exit":
            break

        if user_input.lower() == "nc":
            thread_id += 1
            continue

        try:
            question = user_input
            config = {"configurable": {"thread_id": "abc123"}}

            questions = generate_queries_decomposition.invoke(
                {"question": question, "chat_history": chat_history}
            )

            q_a_pairs = ""
            for q in questions:

                rag_chain = (
                    {
                        "context": itemgetter("question") | retriever,
                        "question": itemgetter("question"),
                        "q_a_pairs": itemgetter("q_a_pairs"),
                    }
                    | decomposition_prompt
                    | llm
                    | StrOutputParser()
                )

                answer = rag_chain.invoke(
                    {"question": q, "q_a_pairs": q_a_pairs}, stream="true"
                )
                q_a_pair = format_qa_pair(q, answer)
                q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair

            print(answer)
            chat_history.extend(
                [
                    HumanMessage(content=question),
                    AIMessage(content=answer),
                ]
            )

            # AGENTIC SOLUTION

            # for s in agent_executor.stream(
            #     {
            #         "messages": [
            #             SystemMessage(
            #                 content="""
            #                 You are a helpful assistant that write code for a user problem, you have tools to retrieve
            #                 relevant documentation and code snippets for particular library and framework based on user queries and subproblems.
            #                 You should try to solve the problem by dividing it into subproblem  based on user query.
            #                 Then for each subproblem, you should use the tools to retrieve the relevant documentation and code snippets.
            #                 After that, you should write the code to solve the subproblem.

            #                 """
            #             ),
            #             HumanMessage(content=query),
            #         ]
            #     },
            #     config=config,
            # ):
            #     print(s)
            #     print("----")

        except Exception as e:
            print(f"Invalid input: {e}")
            continue


# asyncio.run(main())
