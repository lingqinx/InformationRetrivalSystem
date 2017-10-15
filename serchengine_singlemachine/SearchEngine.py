__author__ = 'songmingye'

from flask import Flask, render_template, request

import DocReader as dr
from TextQueryProcessor import TextQueryProcessor

DEBUG = True
SECRET_KEY = 'query'

app = Flask(__name__)
app.config.from_object(__name__)

# text = dr.readFromName('/Users/songmingye/Documents/SearchEngine','MED','\n\n')
text = dr.readFromName('/Users/songmingye/Documents/SearchEngine','AMiner-Author.txt','\n\n')
model = TextQueryProcessor(text,query_type='full',k=100,tol=0.12)

@app.route('/',methods=['GET', 'POST'])
def query():
    if request.method == 'POST':
        query = [request.form['querywords']]
        result = model.querying(query)
        result = result[result > 29]
        contents = [dict(title=row-29, text=model.docs[row]) for row in result]
        return render_template('index.html', contents=contents, keywords=request.form['querywords'])
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
