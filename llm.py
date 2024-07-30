from langchain_community.chat_models import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

# get environment variables
load_dotenv('.env')

# set the llm we want to use
# llm = ChatOllama(model="gemma:2b")
llm = ChatOllama(model="qwen2:0.5b")


# # API
# from langchain_community.llms import Replicate

# # REPLICATE_API_TOKEN = getpass()
# # os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
# replicate_id = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
# llama2_chat_replicate = Replicate(
#     model=replicate_id, input={"temperature": 0.01, "max_length": 500, "top_p": 1}
# )

# connect to database
USERNAME = 'sa'

# use string interpolation to create a connection string variable
connectionString = f"""
    DRIVER={{ODBC Driver 17 for SQL Server}};
    SERVER={os.getenv('SERVER')};
    DATABASE={os.getenv('DATABASE')};
    UID={USERNAME};
    PWD={os.getenv('PASSWORD')};
"""

uri=os.getenv('uri')
db = SQLDatabase.from_uri(uri, sample_rows_in_table_info=0)

# Define functions for schema retrieval and query execution
def get_schema(_):
    return db.get_table_info()


def run_query(query):
    return db.run(query)

# Prompt
template = """Given an input question, convert it to a SQL query. No pre-amble. Based on the table schema below, write a SQL query that would answer the user's question:
{schema}
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

memory = ConversationBufferMemory(return_messages=True)

# Chain to query with memory
from langchain_core.runnables import RunnableLambda

sql_chain = (
    RunnablePassthrough.assign(
        schema=get_schema,
        history=RunnableLambda(lambda x: memory.load_memory_variables(x)["history"]),
    )
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)


def save(input_output):
    output = {"output": input_output.pop("output")}
    memory.save_context(input_output, output)
    return output["output"]


sql_response_memory = RunnablePassthrough.assign(output=sql_chain) | save
# sql_response_memory.invoke({"question": "how many users do i have in my database?"})


# Chain to answer
template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""
prompt_response = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input question and SQL response, convert it to a natural language answer. No pre-amble.",
        ),
        ("human", template),
    ]
)

full_chain = (
    RunnablePassthrough.assign(query=sql_response_memory)
    | RunnablePassthrough.assign(
        schema=get_schema,
        response=lambda x: db.run(x["query"]),
    )
    | prompt_response
    | llm
)

# full_chain.invoke({"question": "how many people in my users table are admins?"})
# print(response)