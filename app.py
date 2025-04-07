from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from dotenv import load_dotenv

st.title('AI Report Generator')
load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    timeout=None,
    max_retries=2,
)


template = (
    "{input}: Detail report"
    "user input is unknown or only special character based simply give error message after the user provided input"
    "You need to generate detailed report based on user input"
    "You need to provide abstract after that introduction"
    "You need to provide main report content in which you need more input based content like functional and non functional requirements but it's optional for some input"
    "You need to provide technology stack it's optional for some input"
    "You need to provide future enhancement it's optional for some input"
    "you don't need to indicate that it's optional"
    "Provide conclusion after that references"
    "In reference avoid optional keyword you must be provide references if possible then provide link of that resources also"
    "You need to do document type formatting like sections font-size is extra larger than subsections"
    "I don't want to see optional keyword in entire report"

    "1. Abstract"
    "2. Introduction"
    "3. Main content"
    "4. Technology stack"
    "5. Future enhancement"
    "6. Conclusion"
    "7. References"

    "Follow above order with provided instructions"
)


prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "{input}"),
])


chain = prompt | llm | StrOutputParser()

query = st.chat_input('Say something')
input_query = query

if input_query:
    response = chain.invoke({"input": input_query})

    st.write(response)
else:
    pass
