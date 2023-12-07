### importing the packages

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
# from langchain.llms import OpenAI
import pinecone
from langchain.chains.question_answering import load_qa_chain

## the streamlit packages
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from utils import *

st.title('Personal Attending 🧑🏾‍⚕️')


if 'responses' not in st.session_state:
    st.session_state['responses'] = ['Hello Doc, how can I assist you today?']


if 'requests' not in st.session_state:
    st.session_state['requests'] = [ ]


## define the chat model

llm = ChatOpenAI(model_name = 'gpt-3.5-turbo', temperature = 0.0)
# llm = OpenAI(temperature= 0.0)

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k =3, return_messages = True)


# Get result from QA chain
# result = qa({"question": query, "chat_history": chat_history_tuples})


## define the system message.

system_msg_template = SystemMessagePromptTemplate.from_template(template= """ Answer the  question as an expert medical professional using the provided context, "
if the answer is not available say I do not know, don't try to make up the answer, remember you are speaking to another healthcare professional  so do not say they need
consult a healthcare professional in your response""")

human_msg_template = HumanMessagePromptTemplate.from_template(template ='{input}')

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])


conversation = ConversationChain(memory = st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# st.title("Langchain Chatbot")
response_container = st.container()

textcontainer = st.container()



with textcontainer:
    query = st.text_input("Type your question in here")
    if query:
        with st.spinner("Refining..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            refined_query = query_refiner(conversation_string, query)
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = find_match(refined_query)
            # print(context)
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

