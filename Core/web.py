import os
import uuid
from flask import Flask, request, send_file
from core import run
app = Flask(__name__)
pwd = os.path.join(os.path.dirname(__file__), 'News')
ALLOWED_EXTENSIONS = {'xlsx', 'csv'}
model = run()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/single', methods=['POST'])
def single():
    title = request.form['title']
    content = request.form['content']
    return model.single(title, content)


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file.filename != '' and allowed_file(file.filename):
        filename = ''.join(str(uuid.uuid4()).split('-')) \
            + "." + file.filename.rsplit('.', 1)[1].lower()
        file.save(os.path.join(pwd, "upload", filename))
        # --------中间执行过程------------ #



        # ------------略----------------- #
        file.seek(0)
        file.save(os.path.join(pwd, "download", filename))
        return filename
    return 'error'


@app.route("/download", methods=['GET'])
def download_file():
    filename = request.args.get('fileId')
    filepath = os.path.join(pwd, 'download', filename)
    if os.path.isfile(filepath):
        return send_file(filepath, as_attachment=True)
    else:
        return 'error'


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=80)

