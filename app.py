import streamlit as st
import pandas as pd
from io import StringIO
import os
from langchain.llms import OpenAI
import pandas as pd
import openai

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate



def textEmbeddings(data):
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="map_reduce", retriever=docsearch.as_retriever())
    prompt_template = """Ignore all previous instructions. You are the world's best interviewer now. I am going to give you a resume of a candidate. Analyze the resume in 4 categories: Education, Work Experience, Projects and Internships, Others including interests, skills etc. Be simple, direct and commanding. Start with greeting the candidate with a 2 line relatable introduction emphasizing your superiority. Ask the candidate if they have a particular company and a role that they want to apply for. 
    If the candidate mentions either the company or the role, then ensure all questions that would be asked will are related to it.
    If they don't mention either the company or role clearly, then ignore this and move to the next step. 
    Then, give a one line response acknowledging the candidate or if they are not clear about the company or the role then acknowledge positively that you would ask practice interview questions. Then ask the candidate topic would they like to start with. There are 4 categories of questions: educational background related, role related or technical questions, behavioral questions and HR or culture related questions. Here, the candidate will have to put an input. 
    Now you will have to ask interview questions. Ensure the questions are good have test the candidate's knowledge. You can choose between longer case based questions, hypothetical questions or academic questions etc. as you deem fit. 
    If the candidate types educational background related, ask it 3-4 most relevant questions related to their education based on their resume which are relevant for the role or the company. 
    If the candidate types role related or technical related then ask accordingly. Here you can even ask a coding question or test their technical understanding etc. 
    Similarly, do it for behavioral questions and HR or culture related questions. You can also be creative, funny, or show emotions at time.
    {context}
    Question: {question}
    Answer in possible questions for interview:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs)
    
    return qa
    
    pass

def ai(category, data):
    qa = textEmbeddings(data)
    query = "please suggest "+ category +" interview questions"
    data = list(filter(None, qa.run(query).split('\n')))
    results = list(filter(lambda x: x != ' ', data))
#     results = ['1. What inspired you to pursue a career in data science?',
#              '2. What experience do you have in predictive modelling, data processing, and data mining algorithms?',
#              '3. What challenges have you faced in building data-intensive applications?',
#              '4. Tell us about your experience with scripting languages such as Python.',
#              '5. How have you used NLP in your past projects?',
#              '6. Describe the process of creating, developing, testing, and deploying diverse services.',
#              '7. What motivated you to develop an NLP powered adaptive Chatbot?',
#              '8. How did you deploy the Joey Chatbot (Text to SQL) project?',
#              '9. What techniques have you used to process, clean, and verify real-time data integrity?',
#              '10. What challenges did you face while mentoring fellowship engineers on machine learning?']
    results = "\n".join(results)
    
    return results
    

st.markdown(
            """
            <h1 style='text-align: center;'>Personalize Interview Questions based on Resume</h1>
            """,
            unsafe_allow_html=True,
        )


uploaded_file = st.file_uploader("Choose a file",type="pdf")
resumeData = None

if uploaded_file is not None:
    
    from langchain.document_loaders import UnstructuredPDFLoader
    loader = UnstructuredPDFLoader(uploaded_file)
    resumeData = loader.load()

category = ["Technical", "Education Background", "Behaviour", "Project Specific"]

for value in category:

    with st.expander(value + " Questions"):
        response  = ai(value, resumeData)
        st.write(response)

