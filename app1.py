import streamlit as st 
from transformers import TextStreamer
from unsloth import FastLanguageModel
import torch
import asyncio 
import asyncpg
from pgvector.asyncpg import register_vector
from google.cloud.sql.connector import Connector
import re

max_seq_length = 2048  
dtype = None 
load_in_4bit = True  

st.title(' 🤖 Genesis.AI ')

items = ["Yuki20/llama3_8b_sql4","Yuki20/llama3_8b_sql5","Yuki20/llama3_8b_sql3","unsloth/llama-2-7b-chat-bnb-4bit" ]

# Title of the sidebar
st.sidebar.title("Model Selection")

# Dropdown menu
selected_item = st.sidebar.selectbox("Select an option", items)
 
sql_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Generate SQL for Question: {}

### Input:
CREATE TABLE patient_record (
                        id INT PRIMARY KEY,
                        transcription TEXT,
                        keywords TEXT ,
                        first_name VARCHAR(1024) ,
                        last_name VARCHAR(1024) ,
                        age INT ,
                        gender VARCHAR(20), 
                        date DATE )

### Response:
{}"""

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

@st.cache_resource
def load_model_and_tokenizer(model_name):

    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer(selected_item)

async def execute_query(query: str):
    # Set your project-specific variables
    project_id = "adsp-capstone-utku-team4"
    region = "us-west2"
    instance_name = "pg15-pgvector-demo"  # replace with your Cloud SQL instance name
    database_user = "admin"  # replace with your database user
    database_password = "521521"  # replace with your database password
    database_name = "patient_record"  # replace with your database name

    loop = asyncio.get_running_loop()
    async with Connector(loop=loop) as connector:
        # Create connection to Cloud SQL database.
        conn: asyncpg.Connection = await connector.connect_async(
            f"{project_id}:{region}:{instance_name}",  # Cloud SQL instance connection name
            "asyncpg",
            user=f"{database_user}",
            password=f"{database_password}",
            db=f"{database_name}",
        )

        await register_vector(conn)  # If this is necessary, make sure to define this function

        results = await conn.fetch(query)

        if len(results) == 0:
            raise Exception("Did not find any results. Adjust the query parameters.")

        await conn.close()
    return results

def run_query_sync(query: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = loop.run_until_complete(execute_query(query))
    return results


# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

def truncate_text(text, max_length=7000):
    if len(text) > max_length:
        return text[:max_length]
    return text

def generate_response_sql(input_text):
    inputs = tokenizer(
    [
        sql_prompt.format(
           input_text, # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    text_streamer = TextStreamer(tokenizer)
    result = model.generate(**inputs, streamer=text_streamer, max_new_tokens=500, use_cache=True)
    return tokenizer.decode(result[0], skip_special_tokens=True)

def generate_response(instruction, context):
    if not context:
        context = "no context"
    else:
        context = truncate_text(context)
    
    inputs = tokenizer(
    [
        alpaca_prompt.format(
           instruction,
           context, # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    text_streamer = TextStreamer(tokenizer)
    result = model.generate(**inputs, streamer=text_streamer, max_new_tokens=500, use_cache=True)
    return tokenizer.decode(result[0], skip_special_tokens=True)

with st.form('my_form'):
    
    text = st.text_area('Ask Question:')
    submitted = st.form_submit_button('Submit') 
    if submitted:
        try:
            # Pass to LLM to generate SQL
            sql_query = generate_response_sql(text)
            # Regular expression to capture the text after 'SELECT'
            pattern = re.compile(r'SELECT.*', re.IGNORECASE | re.DOTALL)

            # Search for the pattern in the text
            match = pattern.search(sql_query)

            if match:
                # Extract the matched text
                sql = match.group(0)
                st.markdown("### Generated SQL:")
                st.info(sql)
                
                try:
                    # Run the SQL query and get results
                    sql_results = run_query_sync(sql)
                    
                    # Check if sql_results is valid
                    if sql_results:
                        st.markdown("### Database Results:")
                        st.info(sql_results)
                    else:
                        st.markdown("### Database Results:")
                        st.info("No results found or query execution failed.")
                    
                    # Run final response generation if sql_results is valid
                    st.markdown("### Final Response:")
                    summary_result = generate_response(text + ' Summarize the given result.', sql_results)
                    st.info(summary_result)
                except Exception as e:
                    st.error(f"An error occurred while running the query: {e}")
            else:
                st.markdown("### Generated SQL:")
                st.info(sql_query)
                st.info("No SQL statement found")
        except Exception as e:
            st.error(f"An error occurred: {e}")
