from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import pandas as pd
import os
from datetime import datetime, date

# load faces
data = load('dataset.npz')
testX_faces = data['arr_2']
# load face embeddings
data = load('embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

clf = GradientBoostingClassifier(n_estimators=100, 
                                 learning_rate=1.0,
                                 max_depth=1, random_state=0)
clf.fit(trainX, trainy)
# K-NN
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(trainX, trainy)

def assure_path_exists(path):
    try:
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
    except:
        print('Error has occured')

def collect_data(std_name, std_course, std_level, std_date, std_time, std_semester, std_session):

    data = {
    'Name':std_name,
    'Course':std_course,
    'Level':std_level,
    'Date':std_date,
    'Time':std_time,
    'Semester': std_semester,
    'Session': std_session
    }
    student = pd.DataFrame(data, columns=["Name", "Course", "Level", "Date", "Time", "Semester", "Session"])
    return student

def attendance(student):
    path = 'Database/'
    assure_path_exists(path)
    attendance = 'Attendance.csv'
    files = os.listdir(path)
    if attendance in files:
      student.to_csv(path + attendance, mode='a', index=False, header=False)
    else:
      student.to_csv(path + attendance, index=False)
        
    return attendance

# test model on a random example from the test dataset
selection = choice([i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection]
random_face_emb = testX[selection]
random_face_class = testy[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])

# prediction for the face
samples = expand_dims(random_face_emb, axis=0)
yhat_class = clf.predict(samples)
yhat_prob = clf.predict_proba(samples)

knn_yhat_class = neigh.predict(samples)
knn_yhat_prob = neigh.predict_proba(samples)

# get name
class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
print('GBC Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print(' GBC Expected: %s' % random_face_name[0])

name = []
course = []
level = []
date_ = []
time = []
semester = []
session = []

std_course = str(input("Enter the course code e.g CPT111: "))
std_level = str(input("Enter the course code e.g 100: "))
std_semester = str(input("Enter the semester e.g First: "))
std_session = str(input("Enter the session e.g 2022/2023: "))

now = datetime.now()
today = date.today()
now_date = today.strftime("%d-%b-%Y")
now_time = now.strftime("%H:%M:%S")

name.append(predict_names[0])
course.append(std_course)
level.append(str(std_level))
date_.append(now_date)
time.append(now_time)
semester.append(std_semester)
session.append(std_session)
df = collect_data(name, course, level, date_, time, semester, session)
att = attendance(df)

# # get name knn
# knn_class_index = knn_yhat_class[0]
# knn_class_probability = knn_yhat_prob[0,knn_class_index] * 100
# knn_predict_names = out_encoder.inverse_transform(knn_yhat_class)
# print('KNN Predicted: %s (%.3f)' % (knn_predict_names[0], knn_class_probability))
# print('KNN Expected: %s' % random_face_name[0])

y_pred = clf.predict(testX)



print('GBC Validation score: ', clf.score(testX, testy), y_pred)
# print('KNN Validation score: ', neigh.score(testX, testy), y_pred)
# print(classification_report(testX, y_pred))

# plot for fun
pyplot.imshow(random_face_pixels)
title = '%s (%.3f)' % (predict_names[0], class_probability)
pyplot.title(title)
pyplot.show()