import asyncio

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.messages import HumanMessage
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

# asyncio.run(crawl_sitemap(sitemap_url="https://ymahlau.github.io/fdtdx/sitemap.xml"))

thread_id = 1

memory = MemorySaver()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


async def main():

    loader = DirectoryLoader(
        "./output/",
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=True,
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
    )
    retriever = vectorstore.as_retriever()

    ### Build retriever tool ###
    tool = create_retriever_tool(
        retriever,
        "docs_retriever",
        "Search the relevant documentation utils which can help to write code snippets based on user query and subproblems",
    )
    tools = [tool]

    agent_executor = create_react_agent(llm, tools, checkpointer=memory)

    while True:
        user_input = input("Question: ")
        if user_input.lower() == "exit":
            break

        if user_input.lower() == "nc":
            thread_id += 1
            continue

        try:
            query = user_input
            config = {"configurable": {"thread_id": "abc123"}}

            for s in agent_executor.stream(
                {"messages": [HumanMessage(content=query)]}, config=config
            ):
                print(s)
                print("----")

        except Exception as e:
            print(f"Invalid input: {e}")
            continue


asyncio.run(main())
