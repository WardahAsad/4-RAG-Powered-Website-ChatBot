from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

#Get Key
load_dotenv()

st.set_page_config(
    page_title="CrumbChat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Inside DataCrumbs")

st.subheader("Web Chat: Talk with Website Data")
# Loading Data
urls = ['https://datacrumbs.org/' , 'https://datacrumbs.org/about-us/' , 'https://datacrumbs.org/our-courses/' , 'https://datacrumbs.org/internship/' , 'https://datacrumbs.org/hall-of-fame/']

loader = UnstructuredURLLoader(urls = urls)

data = loader.load()

#Split the data / chunking method

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

docs = text_splitter.split_documents(data)

#Create VectorStore / number codes

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

#Retrieve docs similar to provided ques

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

#Create Chat AI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5, max_tokens=500)

# Step 1: Initialize
if "messages" not in st.session_state:
    st.session_state.messages = []

# Step 2: Display chat history
# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])
# Step 1: Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Step 2: Display chat history using columns
for msg in st.session_state.messages:
    if msg["role"] == "user":
        col1, col2 = st.columns([3, 1])  # User on the left
        with col1:
            st.markdown("**👤 You:**")
            st.info(msg["content"])
    elif msg["role"] == "assistant":
        col1, col2 = st.columns([1, 3])  # Bot on the right
        with col2:
            st.markdown("**🤖 Bot:**")
            st.success(msg["content"])




#Create chat box
query =  st.chat_input("Ask any question about DataCrumbs Company")

prompt = query

# 🤖 Write instructions for the AI assistant on how to answer questions:

system_prompt = (
    "You are a helpful assistant designed to answer questions from the website"
    "using the provided context. Base your response only on the"
    "information in the context below. If the answer is not found "
    "in the context, respond with 'I don't know because the answer is not given in the provided context.'"
    "Keep your reply brief, no more than three sentences."
    "\n\n\n"
    "{context}"

)

#Create a prompt template
prompt_temp = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        *[(msg["role"], msg["content"]) for msg in st.session_state.messages],
        ("human", "{input}"),
    ]
)


if query:
    # Save user message
    # st.chat_message("user").write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    #RAG pipeline
    question_answer_chain = create_stuff_documents_chain(llm, prompt_temp)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})
    answer = response["answer"]

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Save and show assistant message
    # Show user query on left
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**👤 You:**")
        st.info(query)

    # Show bot response on right
    col1, col2 = st.columns([1, 3])
    with col2:
        st.markdown("**🤖 Bot:**")
        st.success(answer)