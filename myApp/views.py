from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.models import User, auth
from django.contrib import messages
from pandas.core.frame import DataFrame
from .models import Member, Files
from django.conf import settings
from django.http import HttpResponse, Http404
from django.utils.encoding import smart_str


import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import math
import gspread

import mimetypes
import os


# Create your views here.
def index(request):
    context = {
        'member':Member.objects.all()
    }
    
    return render(request, 'index.html', context)

def register(request):
    if request.method == 'POST':
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        username = request.POST['username']
        password = request.POST['password']
        password2 = request.POST['password2']

        if password == password2:
            if User.objects.filter(username=username).exists():
                messages.info(request, 'Email already used')
                return redirect('register')
            else:
                user = User.objects.create_user(first_name=first_name, last_name=last_name, username=username, password=password)
                user.save();
                return redirect('login')
        else:
            messages.info(request, 'Password not the same')
            return redirect('register')
    else:
        return render(request, 'register.html')

def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = auth.authenticate(username=username, password=password)

        if user is not None:
            auth.login(request, user)
            return redirect('/')
        else:
            messages.info(request, 'Credentials invalid')
            return redirect('login')
    else:
        return render(request, 'login.html')

def logout(request):
    auth.logout(request)
    return redirect('/')

def counter(request):
    posts = [1, 2, 3, 4, 5, 'tim', 'tom', 'john']
    return render(request, 'counter.html', {'posts': posts})

def post(request, pk):
    return render(request, 'post.html', {'pk': pk})

def chart(request):
    gc = gspread.service_account(filename='credentials.json')
    ss = gc.open_by_key('1cIdPmEpklbm5AAx91sOswdyZDK-9RDGIeC8zVbxNxwI')
    worksheet = ss.sheet1
    dataframe = pd.DataFrame(worksheet.get_all_records())


    dataframe.to_csv('dataset.csv', index=False)
    df = pd.read_csv('dataset.csv')

    #clean the data
    df = df.drop(columns=['error'])
    df = df.dropna()
    df = df.loc[(df!=0).all(axis=1)] #removes 0s
    df.to_csv('dataset.csv', index=False)

    dataset = pd.DataFrame(df)
    dataset['timestamp'] = pd.to_datetime(dataset.timestamp)
    datasetHead = dataset.head()
    datasetDescribe = dataset.describe()


    #knn regression
    X  = dataset[['open','high','low','volume']]
    y = dataset['close']

    #separate training and testing
    X_train , X_test , y_train , y_test = train_test_split(X ,y , random_state = 0)
        
    #train
    regressor = LinearRegression()
    regressor.fit(X_train,y_train)

    coef = regressor.coef_

    predicted=regressor.predict(X_test)

    dframe=pd.DataFrame(y_test,predicted)
    dfr=pd.DataFrame({'Actual':y_test,'Predicted':predicted})

    #check accuracy
    score = regressor.score(X_test,y_test)
    mae = metrics.mean_absolute_error(y_test,predicted)
    mse = metrics.mean_squared_error(y_test,predicted)
    rmse = math.sqrt(metrics.mean_squared_error(y_test,predicted))

    if request.method == 'POST':
        userOpen = float(request.POST['userOpen'])
        userHigh = float(request.POST['userHigh'])
        userLow = float(request.POST['userLow'])
        userVolume = int(request.POST['userVolume'])
        userParameters = [[userOpen, userHigh, userLow, userVolume]]
        predictedValue = regressor.predict(userParameters)

    else:
        userOpen = ''
        userHigh = ''
        userLow = ''
        userVolume = ''
        predictedValue = ''

    mydick = {
        'dataset': dataset.to_html(),
#        'datasetHead': datasetHead.to_html(),
#        'datasetDescribe': datasetDescribe.to_html(),
        'graph': return_graph(dataset),
#        'coef': coef,
#        'xtest': X_test,
#        'dfr': dfr.to_html(),
#        'score': score,
#        'mae': mae,
#        'mse': mse,
#        'rmse': rmse,
        'userOpen': userOpen,
        'userHigh': userHigh,
        'userLow': userLow,
        'userVolume': userVolume,
        'predictedValue': predictedValue,
    } 
    return render(request, 'chart.html', context=mydick)

def return_graph(dataset):
    fig = plt.figure(figsize=(12,6))
    plt.plot(dataset['open'])

    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)

    data = imgdata.getvalue()
    
    return data

def advanced(request):
    gc = gspread.service_account(filename='credentials.json')
    ss = gc.open_by_key('1cIdPmEpklbm5AAx91sOswdyZDK-9RDGIeC8zVbxNxwI')
    worksheet = ss.sheet1
    dataframe = pd.DataFrame(worksheet.get_all_records())


    dataframe.to_csv('dataset.csv', index=False)
    df = pd.read_csv('dataset.csv')

    #clean the data
    df = df.drop(columns=['error'])
    df = df.dropna()
    df = df.loc[(df!=0).all(axis=1)] #removes 0s
    df.to_csv('dataset.csv', index=False)

    dataset = pd.DataFrame(df)
    dataset['timestamp'] = pd.to_datetime(dataset.timestamp)
    datasetHead = dataset.head()
    datasetDescribe = dataset.describe()


    #knn regression
    X  = dataset[['open','high','low','volume']]
    y = dataset['close']

    #separate training and testing
    X_train , X_test , y_train , y_test = train_test_split(X ,y , random_state = 0)
        
    #train
    regressor = LinearRegression()
    regressor.fit(X_train,y_train)

    coef = regressor.coef_

    predicted=regressor.predict(X_test)

    dframe=pd.DataFrame(y_test,predicted)
    dfr=pd.DataFrame({'Actual':y_test,'Predicted':predicted})

    #check accuracy
    score = regressor.score(X_test,y_test)
    mae = metrics.mean_absolute_error(y_test,predicted)
    mse = metrics.mean_squared_error(y_test,predicted)
    rmse = math.sqrt(metrics.mean_squared_error(y_test,predicted))

    if request.method == 'POST':
        userOpen = float(request.POST['userOpen'])
        userHigh = float(request.POST['userHigh'])
        userLow = float(request.POST['userLow'])
        userVolume = int(request.POST['userVolume'])

    else:
        userOpen = ''
        userHigh = ''
        userLow = ''
        userVolume = ''

    mydick = {
        'dataset': dataset.to_html(),
        'datasetHead': datasetHead.to_html(),
        'datasetDescribe': datasetDescribe.to_html(),
        'graph': return_graph(dataset),
        'coef': coef,
        'xtest': X_test,
        'dfr': dfr.to_html(),
        'score': score,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'userOpen': userOpen,
        'userHigh': userHigh,
        'userLow': userLow,
        'userVolume': userVolume,
    } 
    return render(request, 'advanced.html', context=mydick)

def download_file(request):
    # Define Django project base directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Define text file name
    filename = 'app.apk'
    # Define the full file path
    #filepath = BASE_DIR + '/myApp/Files/' + filename
    
    # Open the file for reading content
    #path = open(filepath, 'rb')
    ## Set the mime type
    #mime_type, _ = mimetypes.guess_type(filepath)
    ## Set the return value of the HttpResponse
    #response = HttpResponse(path, content_type=mime_type)
    ## Set the HTTP header for sending to browser
    #response['Content-Disposition'] = "attachment; filename=%s" % filename
    ## Return the response value
    #return response

    file_path = os.path.join(settings.MEDIA_ROOT, '/myApp/Files/app.apk')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/vnd.android.package-archive")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
            return response
    raise Http404