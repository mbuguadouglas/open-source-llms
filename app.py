from flask import Flask,jsonify,request
from llm import sql_response_memory ,full_chain
from dotenv import load_dotenv
import pyodbc
import os
import uuid

app = Flask(__name__)
load_dotenv('.env')

USERNAME = 'sa'
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# use string interpolation to create a connection string variable
connectionString = f"""
    DRIVER={{ODBC Driver 17 for SQL Server}};
    SERVER={os.getenv('SERVER')};
    DATABASE={os.getenv('DATABASE')};
    UID={USERNAME};
    PWD={os.getenv('PASSWORD')};
"""


# function to remove repetitive writing of queries
def query_db(query, params=()):

    # connect to mssql database
    conn = pyodbc.connect(connectionString)     
    cursor = conn.cursor()      #cursor object to interat with db

    # execute the query given
    cursor.execute(query, params)

    # Get the column names from the cursor description
    columns = [column[0] for column in cursor.description]

    # declare empty list to store query results
    results = []
    for row in cursor.fetchall():
        # zip(columns, row) pairs each column name with the corresponding value in the row
        # dict(zip(columns, row)) creates a dictionary for each row with column names as keys
        results.append(dict(zip(columns, row)))

    # close the db connection
    conn.close()
    return results


# endpoint for chatting with db
@app.route('/chatbot', methods=['GET','POST'])
def chatbot():
   
    # Extract data from JSON payload
    data = request.get_json()
    id = str(uuid.uuid4())
    userId = data.get('userId')
    query = data.get('query')

    # Validate input data
    if not all ([userId,query]):
        return jsonify({"error": "Missing query parameter"}), 400

    try:
        answer = sql_response_memory.invoke({"question":"""{query}"""})
        NLPanswer = full_chain.invoke({"question":"""{query}"""})
        print(f"THIS IS THE RESPONSE {answer}")
        print(f"THIS IS THE RESPONSE {NLPanswer}")

        # # add values to the ai table
        # addChat = """
        # INSERT INTO aiChats (id, userId, query, response) 
        # VALUES (?, ?, ?, ?)
        # """

        # conn = pyodbc.connect(connectionString)
        # cursor = conn.cursor()
        # cursor.execute(addChat, (id, userId, query, answer))
        # conn.commit()
        # conn.close()

        return jsonify({"response": answer}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True)