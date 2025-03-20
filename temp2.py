import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
)

# Initialize memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Initialize conversation chain
if "conversation" not in st.session_state:
    st.session_state.conversation = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        verbose=False,  # Set to True for debugging
    )

st.title("Conversational RAG Application")

# Display chat history
st.subheader("Chat History")
for message in st.session_state.memory.chat_memory.messages:
    if message.type == "human":
        st.write(f"User: {message.content}")
    else:
        st.write(f"Assistant: {message.content}")

# Input field for the user query
query = st.text_input("Enter your question:")

# Button to trigger the RAG pipeline
if st.button("Get Answer"):
    if query:
        # Perform the RAG query using ConversationChain
        response = st.session_state.conversation.run(query)

        # Display the answer
        st.write("Answer:", response)
    else:
        st.warning("Please enter a question.")
