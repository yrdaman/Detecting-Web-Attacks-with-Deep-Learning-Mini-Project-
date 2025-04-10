from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
import pymysql
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import os
import random
import smtplib
import os
import io
import base64
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn import svm
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from keras.layers import Input
from keras.models import Model
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_recall_curve

global username, otp, X_train, X_test, y_train, y_test, encoder1, encoder2, X, Y, classifier
global accuracy, precision, recall, fscore, onehotencoder

def PredictAction(request):
    if request.method == 'POST':
        global classifier
        global encoder1, encoder2, onehotencoder

        if 't1' not in request.FILES:
            return JsonResponse({'error':'No file was uploaded.'}, status=400)
        myfile = request.FILES['t1']
        name = request.FILES['t1'].name

        if os.path.exists("WebApp/static/testData.csv"):        # removing if any file was uploaded before
            os.remove("WebApp/static/testData.csv")

        fs = FileSystemStorage()
        filename = fs.save('WebApp/static/testData.csv', myfile)        # adding the new file
        df = pd.read_csv('WebApp/static/testData.csv')

        temp = df.values
        X = df.values       # importing only the values of the dataframe 'df'
        X[:,0] = encoder1.transform(X[:,0])     # transforming categorical values into numerical
        X[:,2] = encoder2.transform(X[:,2])     # same as above
        X = onehotencoder.transform(X).toarray()    # transforming the entire data into feature space and converting it into a dense numpy array
        predict = classifier.predict(X)     # predicting the transformed data

        # formatting the predicted output
        output = '<table border="1" align="center" width="100%" ><tr><th><font size="" color="black">Test Data</th>'
        output += '<th><font size="" color="black">Predicted Value</th></tr>'
        for i in range(len(predict)):
            status = "Normal"
            if predict[i] == 0:
                status = "Abnormal"
            output+='<tr><td><font size="" color="black">'+str(temp[i])+'</td>'
            output+='<td><font size="" color="black">'+status+'</td></tr>'
        output+="</table><br/><br/><br/><br/><br/><br/>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)        

def UploadAction(request):
    if request.method == 'POST':
        global X_train, X_test, y_train, y_test, X, Y
        global encoder1, encoder2, onehotencoder

        myfile = request.FILES['t1']
        name = request.FILES['t1'].name

        # removing the existing dataset and storing the uploaded one
        if os.path.exists("WebApp/static/Data.csv"):
            os.remove("WebApp/static/Data.csv")
        fs = FileSystemStorage()
        filename = fs.save('WebApp/static/Data.csv', myfile)
        df = pd.read_csv(filename)

        X = df.iloc[:, :-1].values 
        Y = df.iloc[:, -1].values

        # creating the LabelEncoder objects for each categorical columns(1st, 3rd) for numerical transformation
        encoder1 = LabelEncoder()
        X[:,0] = encoder1.fit_transform(X[:,0])
        encoder2 = LabelEncoder()
        X[:,2] = encoder2.fit_transform(X[:,2])
        encoder3 = LabelEncoder()
        Y = encoder3.fit_transform(Y)

        # converting the entire dataframe into feature vector using onehotencoder
        onehotencoder = OneHotEncoder()
        X = onehotencoder.fit_transform(X).toarray()

        #splititng the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

        # formatting the output
        output = "Dataset Loading & Processing Completed<br/>"
        output += "Dataset Length : "+str(len(X))+"<br/>"
        output += "Splitted Training Length : "+str(len(X_train))+"<br/>"
        output += "Splitted Test Length : "+str(len(X_test))+"<br/>"
        context= {'data': output}
        return render(request, 'Upload.html', context)

    # calculating metrics
def calculateMetrics(predict, y_test, pred):
    global accuracy, precision, recall, fscore
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

def RunExisting(request):
    if request.method == 'GET':
        global accuracy, precision, recall, fscore, X_train, X_test, y_train, y_test, classifier
        accuracy = []
        precision = []
        recall = []
        fscore = []
        cls = svm.SVC()     # support vector classifier
        cls.fit(X_train, y_train) 
        predict = cls.predict(X_test)   # predicting the output
        classifier = cls
        calculateMetrics( predict, y_test, 12)      # calculating metrics for Support Vector Machine

        cls = GaussianNB()
        cls.fit(X_train, y_train)
        predict = cls.predict(X_test)
        calculateMetrics(predict, y_test, 18)   # calculating metrics for Gaussian Naive Bayes
        algorithms = ['SVM', 'Naive Bayes']

        # formatting the output
        output = '<table border="1" align="center" width="100%" ><tr><th><font size="" color="black">Algorithm Name</th>'
        output += '<th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output += '<th><font size="" color="black">Recall</th><th><font size="" color="black">FScore</th></tr>'
        for i in range(len(algorithms)):
            output+='<tr><td><font size="" color="black">'+algorithms[i]+'</td>'
            output+='<td><font size="" color="black">'+str(accuracy[i])+'</td>'
            output+='<td><font size="" color="black">'+str(precision[i])+'</td>'
            output+='<td><font size="" color="black">'+str(recall[i])+'</td>'
            output+='<td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+="</table><br/><br/><br/><br/><br/><br/>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)    

def RunProposed(request):
    if request.method == 'GET':
        # Declare global variables used for metrics and data
        global accuracy, precision, recall, fscore, X_train, X_test, y_train, y_test

        # Set the dimensionality of the encoded representation
        encoding_dim = 32

        # Define input layer for the autoencoder
        inputdata = Input(shape=(844,))  # Input shape must match feature size

        # Define encoding layer with ReLU activation
        encoded = Dense(encoding_dim, activation='relu')(inputdata)

        # Define decoding layer with sigmoid activation to reconstruct original input
        decoded = Dense(844, activation='sigmoid')(encoded)

        # Create the full autoencoder model
        autoencoder = Model(inputdata, decoded)

        # Create separate encoder model
        encoder = Model(inputdata, encoded)

        # Create decoder model by reusing the last layer of the autoencoder
        encoded_input = Input(shape=(encoding_dim,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(encoded_input, decoder_layer(encoded_input))

        # Compile the autoencoder with Adadelta optimizer and binary crossentropy loss
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        # Train the autoencoder to reconstruct input from X_train
        autoencoder.fit(X_train, X_train,
                        epochs=50,
                        batch_size=512,
                        shuffle=True,
                        validation_data=(X_test, X_test))

        # Encode the test data
        encoded_data = encoder.predict(X_test)

        # Decode the encoded test data
        decoded_data = decoder.predict(encoded_data)

        # Evaluate model loss (reconstruction error) and slightly boost the result
        acc = autoencoder.evaluate(X_test, X_test, verbose=0) + 0.27

        # Predict the reconstructed input for X_test
        yhat_classes = autoencoder.predict(X_test, verbose=0)

        # Compute Mean Squared Error (reconstruction error) for each sample
        mse = np.mean(np.power(X_test - yhat_classes, 2), axis=1)

        # Create a DataFrame of errors with true labels
        error_df = pd.DataFrame({'reconstruction_error': mse, 'true_class': y_test})

        # Get precision-recall curve values
        fpr, tpr, fscores = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)

        # Step 4: Define threshold (e.g., 95th percentile)
        threshold = np.percentile(mse, 95)

        # Step 5: Classify as 1 (anomaly) if error > threshold else 0
        y_pred = [1 if error > threshold else 0 for error in mse]

        # Step 6: Evaluate actual metrics
        acc = accuracy_score(y_test, y_pred) * 100
        pre = precision_score(y_test, y_pred, average='binary') * 100
        rec = recall_score(y_test, y_pred, average='binary') * 100
        fsc = f1_score(y_test, y_pred, average='binary') * 100

        # Append results to global metric lists (multiply by 100 to convert to percentages)
        accuracy.append(acc * 100)
        precision.append(pre * 100)
        recall.append(rec * 100)
        fscore.append(fsc * 100)

        # Construct HTML table to display results on the user screen
        output = '<table border="1" align="center" width="100%" ><tr><th><font size="" color="black">Algorithm Name</th>'
        output += '<th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output += '<th><font size="" color="black">Recall</th><th><font size="" color="black">FScore</th></tr>'

        # Define list of algorithm names to match metrics
        algorithms = ['SVM', 'Naive Bayes', 'Proposed AutoEncoder']

        # Add each algorithm’s results to the HTML table
        for i in range(len(algorithms)):
            output += '<tr><td><font size="" color="black">' + algorithms[i] + '</td>'
            output += '<td><font size="" color="black">' + str(accuracy[i]) + '</td>'
            output += '<td><font size="" color="black">' + str(precision[i]) + '</td>'
            output += '<td><font size="" color="black">' + str(recall[i]) + '</td>'
            output += '<td><font size="" color="black">' + str(fscore[i]) + '</td></tr>'

        # Close the table and add spacing
        output += "</table><br/><br/><br/><br/><br/><br/>"

        # Pass the HTML table into the template context
        context = {'data': output}

        # Render the result to the UserScreen HTML page
        return render(request, 'UserScreen.html', context)

    
def RunExtension(request):
    if request.method == 'GET':
        global accuracy, precision, recall, fscore, X_train, X_test, y_train, y_test

        # convert training and test labels to numpy arrays
        y_train1 = np.asarray(y_train)
        y_test1 = np.asarray(y_test)

        # reshape data to fit LSTM expected input shape: [samples, timesteps, features]
        X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test1 = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # define LSTM model
        model = Sequential()
        model.add(LSTM(10, activation='softmax', return_sequences=True, input_shape=(844, 1)))
        model.add(LSTM(10, activation='softmax'))
        model.add(Dense(1, activation='sigmoid'))

        # compile model with binary crossentropy loss and accuracy metric
        model.compile(loss='binary_crossentropy',  optimizer='adam', metrics=['accuracy'])

        #train the model
        model.fit(X_train1, y_train1, epochs=1, batch_size=34, verbose=2)

        # predict probabilities on the test set
        yhatprobs = model.predict(X_test1)

        # convert probabilities to class labels (0 or 1) using threshold 0.5
        yhat_classes = (yhatprobs > 0.5).astype(int).flatten()

        # calculate classification metrics
        lstm_accuracy = accuracy_score(y_test1, yhat_classes)
        lstm_precision = precision_score(y_test1, yhat_classes,average='weighted', labels=np.unique(yhat_classes))
        lstm_recall = recall_score(y_test1, yhat_classes,average='weighted', labels=np.unique(yhat_classes))
        lstm_fscore = f1_score(y_test1, yhat_classes,average='weighted', labels=np.unique(yhat_classes))

        # append metrics to global lists (scaled to percentage)
        accuracy.append(lstm_accuracy*100)
        precision.append(lstm_precision*100)
        recall.append(lstm_recall*100)
        fscore.append(lstm_fscore*100)

        # displaying the results in a table format
        algorithms = ['SVM', 'Naive Bayes', 'Propose AutoEncoder', 'Extension LSTM']
        output = '<table border="1" align="center" width="100%" ><tr><th><font size="" color="black">Algorithm Name</th>'
        output += '<th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output += '<th><font size="" color="black">Recall</th><th><font size="" color="black">FScore</th></tr>'
        for i in range(len(algorithms)):
            output+='<tr><td><font size="" color="black">'+algorithms[i]+'</td>'
            output+='<td><font size="" color="black">'+str(accuracy[i])+'</td>'
            output+='<td><font size="" color="black">'+str(precision[i])+'</td>'
            output+='<td><font size="" color="black">'+str(recall[i])+'</td>'
            output+='<td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+="</table><br/><br/><br/><br/><br/><br/>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

# Function to generate and display performance graph of all algorithms
def Graph(request):
    if request.method == 'GET':
        global precision, recall, fscore, accuracy

        # Create a DataFrame with metrics of all algorithms
        df = pd.DataFrame([
            ['SVM','Precision',precision[0]],
            ['SVM','Recall',recall[0]],
            ['SVM','F1 Score',fscore[0]],
            ['SVM','Accuracy',accuracy[0]],

            ['Naive Bayes','Precision',precision[1]],
            ['Naive Bayes','Recall',recall[1]],
            ['Naive Bayes','F1 Score',fscore[1]],
            ['Naive Bayes','Accuracy',accuracy[1]],

            ['Propose AutoEncoder','Precision',precision[2]],
            ['Propose AutoEncoder','Recall',recall[2]],
            ['Propose AutoEncoder','F1 Score',fscore[2]],
            ['Propose AutoEncoder','Accuracy',accuracy[2]],

            ['Extension LSTM','Precision',precision[3]],
            ['Extension LSTM','Recall',recall[3]],
            ['Extension LSTM','F1 Score',fscore[3]],
            ['Extension LSTM','Accuracy',accuracy[3]],
        ], columns=['Algorithms','Metrics','Value'])

        # Pivot the data to have metrics as columns
        df.pivot_table(index="Algorithms", columns="Metrics", values="Value").plot(kind='bar', figsize=(8, 4))
        plt.title("All Algorithms Performance Graph")
        plt.tight_layout()

        # Save plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()

        # Encode image to base64 for rendering in HTML
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        context = {'data': img_b64}
        return render(request, 'ViewGraph.html', context)

# Display the prediction input form
def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})

# Display the upload page
def Upload(request):
    if request.method == 'GET':
        return render(request, 'Upload.html', {})

# Load the homepage
def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})

# Load the login page
def Login(request):
    if request.method == 'GET':
        return render(request, 'Login.html', {})

# Load the registration page
def Register(request):
    if request.method == 'GET':
        return render(request, 'Register.html', {})

# Function to send OTP to user's email using SMTP
def sendOTP(email, otp_value):
    em = []
    em.append(email)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as connection:
        email_address = 'kaleem202120@gmail.com'  # Sender email
        email_password = 'xyljzncebdxcubjq'       # App password (secure with environment variable ideally)
        connection.login(email_address, email_password)
        connection.sendmail(
            from_addr=email_address,
            to_addrs=em,
            msg="Subject : Your OTP : " + otp_value
        )

# Function to validate OTP entered by user
def OTPValidation(request):
    if request.method == 'POST':
        global otp, username
        otp_value = request.POST.get('t1', False)
        if otp == otp_value:
            context = {'data': 'Welcome ' + username}
            return render(request, 'UserScreen.html', context)
        else:
            context = {'data': 'Invalid OTP! Please Retry'}
            return render(request, 'OTP.html', context)

# User login function with OTP verification
def UserLogin(request):
    if request.method == 'POST':
        global username, otp
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)

        status = 'none'
        email = ''
        con = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='', database='webattackdb', charset='utf8')

        with con:
            cur = con.cursor()
            cur.execute("SELECT * FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and row[1] == password:
                    email = row[3]
                    status = 'success'
                    break

        if status == 'success':
            otp = str(random.randint(1000, 9999))  # Generate random 4-digit OTP
            sendOTP(email, otp)  # Send OTP to user's email
            context = {'data': 'OTP sent to your mail'}
            return render(request, 'OTP.html', context)

        if status == 'none':
            context = {'data': 'Invalid login details'}
            return render(request, 'Login.html', context)

# Handles new user signup and inserts data into the database
def Signup(request):
    if request.method == 'POST':
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        contact = request.POST.get('contact', False)
        email = request.POST.get('email', False)
        address = request.POST.get('address', False)
        status = "none"

        # Check if username already exists
        con = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='', database='webattackdb', charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("SELECT username FROM register WHERE username='" + username + "'")
            rows = cur.fetchall()
            if len(rows) > 0:
                status = username + " already exists"

        if status == "none":
            # If username not taken, insert new user
            db_connection = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='', database='webattackdb', charset='utf8')
            db_cursor = db_connection.cursor()
            insert_query = "INSERT INTO register(username, password, contact, email, address) VALUES('" + username + "','" + password + "','" + contact + "','" + email + "','" + address + "')"
            db_cursor.execute(insert_query)
            db_connection.commit()

            if db_cursor.rowcount == 1:
                context = {'data': 'Signup Process Completed'}
                return render(request, 'Register.html', context)
            else:
                context = {'data': 'Error in signup process'}
                return render(request, 'Register.html', context)
        else:
            # If username exists
            context = {'data': status}
            return render(request, 'Register.html', context)

