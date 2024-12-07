import os

import streamlit as st
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.agents.agent_types import AgentType
from langchain.utilities import WikipediaAPIWrapper

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

from streamlit_chat import message
from aux_functions import *

# OpenAI API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Load LLM model
llm = OpenAI()

# UI Initialization
st.title('AI Assistant for Data Science')
st.header('Exploratory Data Analysis')
st.write('Hey, I am your AI Assistant and I am here to help you with your Data Analysis')


# Explaination sidebar
with st.sidebar:
    st.write("**For me to Analyze your Data, please provide me a Data file.**")
    st.caption("*Currently excel file is supported with the extension .xlsx*")
    
    st.divider()

# @st.cache_data
# def steps_eda():
#     steps_eda = llm('What are the steps for EDA')
#     return steps_eda

# @st.cache_data
# def data_science_framing():
#     data_science_framing = llm("Write a couple of paragraphs about the importance of framing a data science problem appropriately")
#     return data_science_framing

# @st.cache_data
# def algorithm_selection():
#     data_science_framing = llm("Write a couple of paragraphs about the importance of considering more than one algorithm when trying to solve a data science problem")
#     return data_science_framing

@st.cache_data
def function_agent():
    st.write("**Data Overview**")
    st.write("The first rows of your dataset look like this:")
    st.write(df.head())

    question = "What are the meaning of each of columns of the dataframe, explain its significance"
    st.write("**Data Cleaning**")
    st.write("These are the columns or features of the data:")
    columns_df = pandas_agent.run(question)
    st.write(columns_df)
    
    st.write("**Checking missing values:**")
    missing_values = pandas_agent.run("How many missing values does each column have in this dataframe? Start the answer with 'There are'")
    st.write(missing_values)

    st.write("**Checking duplicate values:**")
    duplicates = pandas_agent.run("Are there any duplicate values in any of the columns if so where?")
    st.write(duplicates)

    st.write("**Data Summarisation**")
    st.write(df.describe())

    st.write("**Data Correlations**")
    correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
    st.write(correlation_analysis)

    st.write("**Identifying Outliers**")
    outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis ")
    st.write(outliers)
    
    st.write("**Additional feature suggestions**")
    new_features = pandas_agent.run("What new features would be interesting to create?")
    st.write(new_features)

    return 

@st.cache_data
def function_variable():
    st.line_chart(df, y = [user_question])
    
    summary_statistics = pandas_agent.run(f"What are the mean, median, mode, standard deviation, variance, range, quartiles, skewness and kurtosis of {user_question}, list them in bullet points like a list")
    st.write(summary_statistics)

    normality = pandas_agent.run(f"Check for normality or specific distribution shapes of {user_question}")
    st.write(normality)
    
    outliers = pandas_agent.run(f"Assess the presence of outliers of {user_question}")
    st.write(outliers)

    trends = pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_question}")    
    st.write(trends)
    
    missing_values = pandas_agent.run(f"Determine the extent of missing values of {user_question}")
    st.write(missing_values)
    return 

@st.cache_data
def function_question_dataframe():
    dataframe_info = pandas_agent.run(user_question_dataframe)
    st.write(dataframe_info)
    return 

@st.cache_resource
def wiki(prompt):
    wiki_research = WikipediaAPIWrapper().run(prompt)
    return wiki_research

@st.cache_data
def prompt_templates():
    data_problem_template = PromptTemplate(
    input_variables=["business_problem"],
    template="Convert the following business problem into a data science problem: {business_problem}."
    )

    model_selection_template = PromptTemplate(
        input_variables=["data_problem", 'wikipedia_research'],
        template="Give me a list of machine learning algorithms that are suitable to solve this problem: {data_problem}, while using this Wikipedia research; {wikipedia_research}"
    )

    return data_problem_template, model_selection_template

@st.cache_resource
def chains():
    data_problem_chain = LLMChain(llm=llm, prompt= prompt_templates()[0], verbose = True, output_key = 'data_problem')

    model_selection_chain = LLMChain(llm=llm, prompt= prompt_templates()[1], verbose = True, output_key = 'model_selection')

    sequential_chain = SequentialChain(chains=[data_problem_chain, model_selection_chain], input_variables=["business_problem", 'wikipedia_research'], output_variables = ['data_problem', 'model_selection'], verbose = True)

    return sequential_chain

@st.cache_data
def chains_output(prompt, wiki_research):
    my_chain = chains()
    my_chain_output = my_chain({'business_problem': prompt, 
                                'wikipedia_research': wiki_research})
    my_data_problem = my_chain_output["data_problem"]
    my_model_selection = my_chain_output["model_selection"]
    
    return my_data_problem, my_model_selection

@st.cache_data
def list_to_selectbox(my_model_selection_input):
    algorithm_lines = my_model_selection_input.split('\n')
    algorithms = [algorithm.split(":")[-1].split('.')[-1].strip() for algorithm in algorithm_lines if algorithm.strip()]
    algorithms.insert(0, "Select Algorithm")
    formatted_list_output = [f"{algorithm}" for algorithm in algorithms if algorithm]
    return formatted_list_output

@st.cache_resource
def python_agent():
    agent_executor = create_python_agent(
        llm = llm,
        tool = PythonREPLTool(),
        verbose = True,
        agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors = True,
        max_iterations = None,
        max_execution_time = 600
    )

    return agent_executor

@st.cache_data
def python_solution(my_data_problem, selected_algorithm, user_xlsx):
    solution = python_agent().run(f" Write a Python Script to solve this: {my_data_problem}, using this algorithm: {selected_algorithm}, using this as your dataset: {user_xlsx}")

    return solution

tab1, tab2 = st.tabs(["Data Analysis and Data Science", "ChatBot"])

with tab1:
    user_xlsx = st.file_uploader("Upload your file Excel file here", type="xlsx")

    # with st.sidebar:
    #     with st.expander('What are the steps for EDA'):
    #         st.write(steps_eda())

    # with st.sidebar:
    #     with st.expander("The importance of framing a data science problem approriately"):
    #         st.caption(data_science_framing())

    # with st.sidebar:
    #     with st.expander("Is one algorithm enough?"):
    #         st.caption(algorithm_selection())


    if user_xlsx is not None:
        # Setting File Pointer to start of the file
        user_xlsx.seek(0)
        df = pd.read_excel(user_xlsx, sheet_name=0)
        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
        
        function_agent()
        
        st.subheader("Variable of study")
        user_question = st.text_input("What variable are you interested in knowing more about?")

        if user_question is not None and user_question != "":
            function_variable()

            st.subheader("Further Study")
        
            user_question_dataframe = st.text_input("Is there anything else you would like to know about the dataframe")
            st.write("Example: What is the max/min value of a variable or what is the best plots to analyze the data")

            if user_question_dataframe is not None and user_question_dataframe not in ("", "no", "No"):
                function_question_dataframe()
                
                st.divider()

                st.header("Business Analytics using Data Science")
                st.write("Now after we have a solid grasp of the data, we intend to use data science to make predictions and find answers to business problems.")

                prompt = st.text_area("What is the business problem you would like to solve")

                if prompt:
                    wiki_research = wiki(prompt)
                    
                    my_data_problem = chains_output(prompt, wiki_research)[0]
                    my_model_selection = chains_output(prompt, wiki_research)[1]

                    st.write(my_data_problem)
                    st.write(my_model_selection)

                    # formatted_list = list_to_selectbox(my_model_selection)
                    # selected_algorithm = st.selectbox("Select Machine Learning Algorithm", formatted_list)
                    
                    # if selected_algorithm is not None and selected_algorithm != "Select Algorithm":
                    #     st.subheader("Solution")
                    #     solution = python_solution(my_data_problem, selected_algorithm, user_xlsx)
                    #     st.write(solution)

with tab2:
    st.subheader("ChatBot")
    st.write("Welcome to the AI Assistant ChatBot!")
    st.write("Got any questions about your data science problem or need help navigating the intricacies of your project? Just type in your queries, and let's unravel the mysteries of your data together! 🔍💻")

    st.write("")

    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["How can I assist you?"]

    if 'requests' not in st.session_state:
        st.session_state['requests'] = []
    
    llm = ChatOpenAI(model_name="gpt-4o-mini")

    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


    system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")

    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

    prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

    conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

    response_container = st.container()
    textcontainer = st.container()


    with textcontainer:
        query = st.text_input("Hello! How can I help you? ", key="input")
        if query:
            with st.spinner("thinking..."):
                conversation_string = get_conversation_string()
                refined_query = query_refiner(conversation_string, query)
                st.subheader("Refined Query:")
                st.write(refined_query)
                context = find_match(refined_query)
                response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
                st.session_state.requests.append(query)
                st.session_state.responses.append(response)

    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i],key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')



    





