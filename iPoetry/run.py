from flaskblog import app

if __name__ == '__main__':
    app.run(debug=False)
    app.run(host="192.168.1.40",port=5000)
