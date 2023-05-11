from flask import Flask, render_template, request, redirect, url_for, session
from utils.video_processing import load_transcript, split_transcript
from utils.database import create_db  
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from utils.prompts import CONDENSE_PROMPT, QA_PROMPT
import os

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", default="fallback_key")  # Use environment variable for secret key

# Initialize outside the request functions for better performance
chat = ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo')
question_generator = LLMChain(llm=chat,  prompt=CONDENSE_PROMPT)
doc_chain = load_qa_chain(chat, prompt=QA_PROMPT, chain_type="stuff")

@app.route('/', methods=['GET', 'POST'])
def home():
    """Home route. Handles form submission for YouTube video URL."""
    try:
        if request.method == 'POST':
            video_url = request.form.get('video_url')
            session['video_url'] = video_url
            return redirect(url_for('chat'))
        return render_template('index.html')
    except Exception as e:
        # Handle exceptions and show a friendly error message
        return render_template('error.html', message=str(e))

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    """Chat route. Handles form submission for user question and generates chatbot response."""
    try:
        if 'video_url' not in session:
            return redirect(url_for('home'))

        video_url = session['video_url']
        transcript = load_transcript(video_url)
        docs = split_transcript(transcript)
        db = create_db(docs)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        chain = ConversationalRetrievalChain(retriever=db, question_generator=question_generator, combine_docs_chain=doc_chain)
        
        if request.method == 'POST':
            user_question = request.form.get('user_question')
            chat_history = session.get('chat_history', [])
            chat_history.append(('User', user_question))
            session['chat_history'] = chat_history
            response = chain.run({"question": user_question, "chat_history": chat_history})
            chat_history.append(('Chatbot', response))
            session['chat_history'] = chat_history

        chat_history = session.get('chat_history', [])
        return render_template('chat.html', chat_history=chat_history)
    except Exception as e:
        # Handle exceptions and show a friendly error message
        return render_template('error.html', message=str(e))

@app.route('/delete-chat-history', methods=['POST'])
def delete_chat_history():
    """Delete chat history route. Clears the chat history in the session."""
    session.pop('chat_history', None)
    return redirect(url_for('chat'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
