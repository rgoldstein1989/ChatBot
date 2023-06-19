from flask import Flask, render_template,jsonify,request
from flask_cors import CORS
import requests,openai
import openai
from langchain.llms import AzureOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_csv_agent
import pandas as pd
from flask import Flask, url_for, render_template, request, redirect, session
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
# app.config['SECRET_KEY'] = 'thisisasecretkey'                           #   for deploy demand
# app.config['DEBUG'] = True
# db = SQLAlchemy(app)
db = SQLAlchemy()
db.init_app(app)
CORS(app)

app.app_context().push()

# load_dotenv()
# API = os.environ['API']

# Load the CSV file as a Pandas dataframe.
# df = pd.read_csv('./dataset.csv')


OPENAI_API_KEY = "1d47812ac75d48cc990233bdb7f27731"
OPENAI_DEPLOYMENT_NAME = "ITsavvy"
MODEL_NAME = "text-davinci-003"
openai.api_type = "azure"
openai.api_base = "https://itsavvy.openai.azure.com/"
openai.api_version = "2023-05-15"
openai.api_key = "1d47812ac75d48cc990233bdb7f27731"
deployment_name='ITsavvy'
openai_api_version="2023-05-15"

# app = Flask(__name__)

# OpenAI API Key
openai.api_key = '1d47812ac75d48cc990233bdb7f27731'


class NewAzureOpenAI(AzureOpenAI):
    stop: list[str] = None
    @property
    def _invocation_params(self):
        params = super()._invocation_params
        # fix InvalidRequestError: logprobs, best_of and echo parameters are not available on gpt-35-turbo model.
        params.pop('logprobs', None)
        params.pop('best_of', None)
        params.pop('echo', None)
        #params['stop'] = self.stop
        return params


# Initiate a connection to the LLM from Azure OpenAI Service via LangChain.
llm = NewAzureOpenAI(
    openai_api_key=OPENAI_API_KEY,
    deployment_name=OPENAI_DEPLOYMENT_NAME,
    model_name=OPENAI_DEPLOYMENT_NAME,
)

# Fine-tune the model on your data
# llm.fit(df['input'], df['output'])

# Save the model and external data using joblib
# joblib.dump((llm, df), "model_data.joblib")

# Load the model and external data from the saved file
# llm, df = joblib.load("model_data.joblib")

agent = create_csv_agent(llm, 'dataset.csv', verbose=True)

def send_message(messages, model_name, max_response_tokens=500):
    response = openai.ChatCompletion.create(
        engine=OPENAI_DEPLOYMENT_NAME,
        messages=messages,
        temperature=0.5,
        max_tokens=max_response_tokens,
        top_p=0.9,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response['choices'][0]['message']['content']

# joblib.dump(agent, "agent.joblib")

# Load the agent
# agent = joblib.load("agent.joblib")

def get_user_input():
    user_input = input("Please enter your question (type 'exit' to quit): ")
    return user_input

def is_related_to_dataset(user_input):
    # Define a list of keywords related to the dataset.csv contents
    keywords = ['billing date','customers', 'customer', 'subscriptions', 'pay', 'subscription', 'license', 'client', 'service name','organization name','customer number','product id','product name','item code','quantity','unit cost price','unit sales price','total cost','total sales','margin amount','margin percentage','currency code','payment frequency','purchase order number','subscription id','subscription name','billing line details'',organization currency code','organization groups','opportunity number']

    # Check if any of the keywords are present in the user_input
    for keyword in keywords:
        if keyword in user_input.lower():
            return True

    return False

def get_completion(user_input):
    # print(prompt)	
    result = agent(user_input)
    output_str = result['output']
    output_str = str(output_str)
    if '{' in output_str:
        output_str = output_str.replace('{','')
    if '}' in output_str:
        output_str = output_str.replace('}','')    
    if '[' in output_str:
        output_str = output_str.replace('[','')
    if ']' in output_str:
        output_str = output_str.replace(']','')
    if '\n' in output_str:
        output_str = output_str.replace('\n',',')
    if '<|im_end|>' in output_str:
        output_str = output_str.replace('<|im_end|>','')
    if '<|im_sep|>' in output_str:
        output_str = output_str.replace('<|im_sep|>','')
    else:
        output_str = output_str
    # output_str = output_str.replace('\n',',').replace('<|im_end|>','')
    # output_str = output_str.split("' '")
    # response_0 = str(result)
    # response = ast.literal_eval(response_0)
    # response = str(response)

    # output_str = response['output']
    # output_str = output_str.replace('[','').replace(']','').replace('\n','').replace('<|im_end|>','')
    # subscription_names = output_str.split("' '")

    # Get the subscriptions by finding the portion of the string square brackets
    # subscription_str = response.split('[')[1].split(']')[0]

    # Split the string into individual subscription names and remove any unwanted characters
    # subscriptions = [s.replace("'", "").strip() for s in subscription_str.split('\n')]

    # Split the string into individual subscription names
    # response = tuple(subscription_str)
    # response = subscription_str
    # response = [x.strip().replace("'", "") for x in response_0.lstrip('[').rstrip(']').split(',')]

    # Remove duplicates by converting the list into a set and back to a list
    # response = list(set(subscriptions))
    return output_str

def generate_response(user_input, messages):
    if is_related_to_dataset(user_input):
        response = get_completion(user_input)
    else:
        messages=[
            {"role": "user", "name":"example_user", "content": user_input}
        ]
        max_response_tokens = 500

        response = send_message(messages, MODEL_NAME, max_response_tokens)
    
    if not response:
        response = "I'm sorry, I couldn't find an answer to your question. Please try rephrasing your question." 
    messages.append({"role": "assistant", "content": response})

    return messages

def print_conversation(messages):
    for message in messages:
         # print(f"[{message['role'].upper()}]")
         print(message['content'])
    return message['content']
     

# messages = []


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self, username, password):
        self.username = username
        self.password = password


@app.route('/', methods=['GET'])
def index():
    if session.get('logged_in'):
        return render_template('home.html')
    else:
        return render_template('index.html', message="Hello, Welcome to ITsavvy Autochat Bot!")


@app.route('/register/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            db.session.add(User(username=request.form['username'], password=request.form['password']))
            db.session.commit()
            return redirect(url_for('login'))
        except:
            return render_template('index.html', message="User Already Exists")
    else:
        return render_template('register.html')


@app.route('/login/', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        u = request.form['username']
        p = request.form['password']
        data = User.query.filter_by(username=u, password=p).first()
        if data is not None:
            session['logged_in'] = True
            return redirect(url_for('index'))
        return render_template('index.html', message="Incorrect Details")


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session['logged_in'] = False
    return redirect(url_for('index'))


@app.route('/data', methods=['POST'])
def get_data():
    
    data = request.get_json()
    text=data.get('data')
    
    # user_input = text
    user_input = str(text)
    # print(user_input)
    try:
        if "licenses" in user_input:
            user_input = user_input.replace('licenses','subscriptions')
        if "licences" in user_input:
            user_input = user_input.replace('licences','subscriptions')
        response = generate_response(user_input, messages = [])
        # response = str(response)
        response = print_conversation(response)
        # model_reply = response
        # print(model_reply)
        return jsonify({"response":True,"message":response})
    except Exception as e:
        print(e)
        error_message = f'Error: {str(e)}'
        return jsonify({"message":error_message,"response":False})

    

# if __name__ == '__main__':
#     app.run()


if(__name__ == '__main__'):
    app.secret_key = "ThisIsNotASecret:p"
    db.create_all()
    app.run()












# @app.route('/')
# def index():
#     return render_template('index.html')

