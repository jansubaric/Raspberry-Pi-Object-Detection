import cv2
import sqlite3
from flask import Flask, Response,render_template, request
from detect import ObjectDetection

 
app = Flask(__name__)
app.debug = True
 
conn = sqlite3.connect('database.db')
print('Opened database successfully!')
 
conn.execute('CREATE TABLE IF NOT EXISTS objects (id INTEGER PRIMARY KEY AUTOINCREMENT, object_name TEXT)')
print('Create table successfully')
conn.close()

object_detection = ObjectDetection()
#camera = cv2.VideoCapture(0)
 
#def gen_frames():
    #while True:
        #success, frame = camera.read()
        #if not success:
            #break
 
        #ret, buffer = cv2.imencode('.jpg', frame)
        #frame = buffer.tobytes()
        #yield(b'--frame\r\n'
              #b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
 
#@app.route('/receive_data', methods=['POST'])
#def receive_data():
    #data = request.json
    #print(data)
    #return 'Data received successfully'
 
#@app.route('/object-detection', methods = ['POST'])
#def object_detection():
    #data = request.form['data']
    #return render_template('result.html', data = data)
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/archive')
def archive():
    return render_template('archive.html')
 
@app.route('/objects', methods = ['POST'])
def object_detection():
    #if request.method == 'POST':
        #data = request.json
        #return render_template('objects.html', data=data)
    #else:
        #return render_template('objects.html')
 
    data = request.json
    print(data)
    try:
         with sqlite3.connect("database.db") as con:
            cur = con.cursor()
            cur.execute("INSERT INTO objects (name) VALUES (?)",("BLABLA") )
 
            con.commit()
            msg = "Record successfully added"
            return "SUCCESSFULLY"
    except:
        con.rollback()
        msg = "error in insert operation"
 
    finally:
        #return render_template("result.html",msg = msg)
        con.close()
        return "FINALLY"
 
@app.route('/show', methods = ['GET'])
def object_detection_get():
    conn = sqlite3.connect("database.db")
 
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM objects')
    tables = cursor.fetchall()
 
    #for table in tables:
        #print(table[0])
 
    return render_template('objects.html', data=tables)
 
 
def gen_frames():
    while True:
        frame = object_detection.get_frame_data()
        if frame is not None:
            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            print('Frame is NONE')
 
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype = 'multipart/x-mixed-replace; boundary=frame')
 
if __name__ == '__main__':
    object_detection.main()
    app.run(host='0.0.0.0', port = 5000)
 