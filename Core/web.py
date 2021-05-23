from flask import Flask
from flask import request
from core import run
app = Flask(__name__)
model = run()


@app.route('/single', methods=['POST'])
def single():
    title = request.form['title']
    content = request.form['content']
    return model.single(title, content)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=80)


