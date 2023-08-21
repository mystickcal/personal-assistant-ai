
from flask import Flask, render_template, request, jsonify
import chat

app = Flask(__name__)

@app.route('/')
def home():
    user_name = request.cookies.get('userName', 'User')
    return render_template('index.html', user_name=user_name)

@app.route('/send_message', methods=['POST'])
def chat_endpoint():
    user_input = request.json['message']
    user_name = request.json['user_name']
    bot_message = chat.chat_with_aime(user_input)
    response_text = {'bot_message': bot_message, 'user_name': user_name}
    return jsonify(response_text)

if __name__ == '__main__':
    app.run(debug=True)
