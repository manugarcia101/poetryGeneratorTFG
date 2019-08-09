from flask import render_template, url_for, flash, redirect, request
from flaskblog import app, bcrypt, mongo
from flaskblog.models import insertPoem, deletePoem, findPoem, findAll
from flaskblog.RNN.char import Char_LSTM
from flaskblog.RNN.char1 import Char_LSTM1
from flaskblog.RNN.word_lstm_generate import Word_LSTM
import tensorflow as tf
import datetime
import re
import string
import pickle

modelGenerator = Word_LSTM()
charGenerator = Char_LSTM()
charGenerator1 = Char_LSTM1()

graph = tf.get_default_graph()

@app.route("/")
@app.route("/iPoetry")
def iPoetry():   
    # processBDExport()
    # processBDImport()
    return render_template('layoutMain.html')

@app.route("/eng_home")
def eng_home():    
    return render_template('eng_layoutMain.html')

@app.route("/poem", methods=['POST'])
def poem():
    global modelGenerator
    global graph
    with graph.as_default():
        _seed = request.form.get('seed')
        _author = request.form.get('author')
        if _author == '':
            _author = "Anónimo"
        _title = request.form.get('poemtitle')
        if _title == '':
            _title = "Anónimo"
        _escritor = request.form.get('escritor')
        if _escritor == '':
            _escritor = "juliocortázar"
        modelGenerator.setWriter(_escritor)
        _poem_generated = modelGenerator.gen_poem(_seed.lower())

        _poem_generated_processed = processPoem(_poem_generated)

    date = datetime.datetime.now()

    Poem = {"Poem":_poem_generated, "Author":_author, "Title":_title, "Fecha":date}

    insertPoem(Poem)

    return render_template('poem.html', poem=_poem_generated_processed, author=[_author], poemTitle=[_title], escritor=[_escritor])

@app.route("/eng_poem", methods=['POST'])
def eng_poem():
    global modelGenerator
    global graph
    with graph.as_default():
        _seed = request.form.get('seed')
        _author = request.form.get('author')
        if _author == '':
            _author = "Anonimous"
        _title = request.form.get('poemtitle')
        if _title == '':
            _title = "Anonimous"
        if _escritor == '':
            _escritor = "juliocortázar"
        _escritor = request.form.get('escritor')
        modelGenerator.setWriter(_escritor)
        _poem_generated = modelGenerator.gen_poem(_seed.lower())

        _poem_generated_processed = processPoem(_poem_generated)

    date = datetime.datetime.now()

    Poem = {"Poem":_poem_generated, "Author":_author, "Title":_title, "Fecha":date}

    insertPoem(Poem)

    return render_template('eng_poem.html', poem=_poem_generated_processed, author=[_author], poemTitle=[_title], escritor=[_escritor])

@app.route("/generatePoem")
def generatePoem():
    return render_template('generatePoem.html')

@app.route("/eng_generatePoem")
def eng_generatePoem():
    return render_template('eng_generatePoem.html')

@app.route("/searchPoem", methods=['POST'])
def searchPoem():
    Poem = request.form.get('title_field')
    search = findPoem(Poem)
    for i in range(len(search)):
        search[i]["Poem"] = processPoem(search[i]["Poem"])
    return render_template('searchPoem.html', search=search)   

@app.route("/eng_searchPoem", methods=['POST'])
def eng_searchPoem():
    Poem = request.form.get('title_field')
    search = findPoem(Poem)
    for i in range(len(search)):
        search[i]["Poem"] = processPoem(search[i]["Poem"])
    return render_template('eng_searchPoem.html', search=search)   

@app.route("/publicPoem")
def publicPoem():
    search = findAll()
    for i in range(len(search)):
        search[i]["Poem"] = processPoem(search[i]["Poem"])
        
    return render_template('publicPoem.html', search=search)   

@app.route("/eng_publicPoem")
def eng_publicPoem():
    search = findAll()
    for i in range(len(search)):
        search[i]["Poem"] = processPoem(search[i]["Poem"])
        
    return render_template('eng_publicPoem.html', search=search)   

@app.route("/testtheRNN")
def testtheRNN():
    return render_template('testtheRNN.html')   

@app.route("/eng_testtheRNN")
def eng_testtheRNN():        
    return render_template('eng_testtheRNN.html')  

@app.route("/RNN1", methods=['POST'])
def RNN1():
    global charGenerator
    global graph

    with graph.as_default():
        _seed = ""
        _author = "Anonimo"
        _title = "Anonimo"
        poema = charGenerator.gen_poem(_seed.lower())

    return render_template('poem.html', poem=poema, author=[_author], poemTitle=[_title])   

@app.route("/RNN2", methods=['POST'])
def RNN2():        
    global charGenerator1
    global graph

    with graph.as_default():
        _seed = ""
        _author = "Anonimo"
        _title = "Anonimo"
        poema = charGenerator1.gen_poem(_seed.lower())

    return render_template('poem.html', poem=poema, author=[_author], poemTitle=[_title])   

@app.route("/RNN3", methods=['POST'])
def RNN3():
    global modelGenerator
    global graph

    with graph.as_default():
        _seed = ""
        _author = "Anonimo"
        _title = "Anonimo"
        _escritor = "antoniomachado"
        modelGenerator.setWriter(_escritor)
        _poem_generated = modelGenerator.gen_poem(_seed.lower())
        _poem_generated_processed = processPoem(_poem_generated)

    return render_template('poem.html', poem=_poem_generated_processed, author=[_author], poemTitle=[_title], escritor=[_escritor])  

@app.route("/RNN4", methods=['POST'])
def RNN4(): 
    global modelGenerator
    global graph

    with graph.as_default():
        _seed = ""
        _author = "Anonimo"
        _title = "Anonimo"
        _escritor = "antoniomachado1"
        modelGenerator.setWriter(_escritor)
        _poem_generated = modelGenerator.gen_poem(_seed.lower())
        _poem_generated_processed = processPoem(_poem_generated)
        _escritor = "antoniomachado"

    return render_template('poem.html', poem=_poem_generated_processed, author=[_author], poemTitle=[_title], escritor=[_escritor])  

@app.route("/eng_RNN1", methods=['POST'])
def eng_RNN1():
    global charGenerator

    _seed = ""
    _author = "Anonimo"
    _title = "Anonimo"

    return render_template('eng_poem.html', poem=charGenerator.gen_poem(_seed.lower()), author=[_author], poemTitle=[_title])   

@app.route("/eng_RNN2", methods=['POST'])
def eng_RNN2():        
    global charGenerator1

    _seed = ""
    _author = "Anonimo"
    _title = "Anonimo"

    return render_template('eng_poem.html', poem=charGenerator1.gen_poem(_seed.lower()), author=[_author], poemTitle=[_title])   

@app.route("/eng_RNN3", methods=['POST'])
def eng_RNN3():
    global modelGenerator
    global graph

    with graph.as_default():
        _seed = ""
        _author = "Anonimo"
        _title = "Anonimo"
        _escritor = "antoniomachado"
        modelGenerator.setWriter(_escritor)
        _poem_generated = modelGenerator.gen_poem(_seed.lower())
        _poem_generated_processed = processPoem(_poem_generated)

    return render_template('eng_poem.html', poem=_poem_generated_processed, author=[_author], poemTitle=[_title], escritor=[_escritor])  

@app.route("/eng_RNN4", methods=['POST'])
def eng_RNN4(): 
    global modelGenerator
    global graph

    with graph.as_default():
        _seed = ""
        _author = "Anonimo"
        _title = "Anonimo"
        _escritor = "antoniomachado1"
        modelGenerator.setWriter(_escritor)
        _poem_generated = modelGenerator.gen_poem(_seed.lower())
        _poem_generated_processed = processPoem(_poem_generated)
        _escritor = "antoniomachado"

    return render_template('eng_poem.html', poem=_poem_generated_processed, author=[_author], poemTitle=[_title], escritor=[_escritor])  

@app.errorhandler(403)
@app.errorhandler(405)
def error403(error):
    return render_template('error403.html')  

@app.errorhandler(404)
def error404(error):
    return render_template('error404.html')  

@app.errorhandler(500)
def error500(error):
    return render_template('error500.html')   

def processPoem(_poem_generated):
    _poem_generated_processed = []
    for item in _poem_generated.split(' '):
        words = item.strip(string.punctuation + '—¡¿“”').split('\n')
        for i in range(len(words)):
            if words[i] != '' and words[i] != 'i' and words[i] != 'x' and words[i] != 'v' and words[i] != 'ii' and words[i] != 'iii' and words[i] != 'vi' and words[i] != 'xi' and words[i] != 'vii' and words[i] != 'xii' and words[i] != 'xiii' and words[i] != 'viii' and words[i] != 'ix' and words[i] != 'iv' and words[i] != 'xx' and words[i] != 'xxi' and words[i] != 'xix':
                _poem_generated_processed.append(words[i])
            if i > 0 and i != len(words) - 1 and len(_poem_generated_processed) > 0:
                if _poem_generated_processed[-1] != '\n':
                    _poem_generated_processed.append('\n')
                    _poem_generated_processed.append('\n')

    return _poem_generated_processed

def processBDExport():
    search = findAll()
    pickle.dump( search, open( "poemdb.p", "wb" ) ) 

def processBDImport():
    poems = pickle.load( open( "poemdb.p", "rb" ) )
    for i in range(len(poems)):
        insertPoem(poems[i])