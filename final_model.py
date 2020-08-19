from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
from flask_bootstrap import Bootstrap
import os
import sys
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
app = Flask(__name__)

teamDict = {
		  1 : "Man United",
		  2 : "Leicester",
		  3 : "Bournemouth",
		  4 : "Cardiff",
		  5 : "Fulham",
		  6 : "Crystal Palace",
		  7 : "Huddersfield",
		  9 : "Chelsea",
		  10 : "Newcastle",
		  11 : "Tottenham",
		  12 : "Watford",
		  13 : "Brighton",
		  14 : "Everton",
		  15 : "Arsenal",
		  16 : "Man City",
		  17 : "Liverpool",
		  18 : "West Ham",
		  19 : "Southampton",
		  20 : "Burnley",
		  21 : "West Brom",
		  22 : "Stoke",
		  23 : "Swansea"
		}

@app.route('/')
def home():
	return render_template('index.html')


@app.route('/ftbl_dashboard')
def ftbl_dashboard():
	return render_template('ftbl_dashboard.html')
	

@app.route('/ftb_pred')
def ftb_pred():
	return render_template('ftb_pred.html')

@app.route('/ftbl_ranking')
def ftbl_ranking():
	return render_template('ftbl_ranking.html')


@app.route('/ftbl_videos')
def ftbl_videos():
	return render_template('ftbl_videos.html')

@app.route('/ftbl_schedule')
def ftbl_schedule():
	return render_template('ftbl_schedule.html')

@app.route('/footballnews')
def footballnews():
	return render_template('footballnews.html')


@app.route('/predict', methods=['POST'])
def predict():
	
	
	df = pd.read_csv("data/football_final.csv")
	#print(df)
	X = np.array(df.drop(['Result'],axis = 1))
	y = np.array(df['Result'])


	
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


	if request.method == 'POST' :
		
		TeamA = int( request.form['TeamA'])
		
		TeamB = int(request.form['TeamB'])
		has =float( request.form['has'])
		hds = float(request.form['hds'])
		aas =float( request.form['aas'])
		ads = float(request.form['ads'])
		algo = float(request.form['algo'])

		list = [[TeamA,TeamB,has,hds,aas,ads]]
		finalList = [[teamDict[TeamA],teamDict[TeamB],has,hds,aas,ads]]
		print('')
		print(list)
		print('')

		

		#data=[[40,	1,	18,	1,	0,	0,	0,	1,	0.25	,0,	0,	1,	2,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	1,	2,	2,	0,	0,	0,	0,	1,	1,	1]]
		if(algo==1):
			from sklearn.neighbors import KNeighborsClassifier
			classifier1 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
			classifier1.fit(X_train, y_train)
			my_prediction = classifier1.predict(list) 
			print("KNearestNeighbour")
		else:
			from sklearn.ensemble import RandomForestClassifier
			classifier2 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0) 
			classifier2.fit(X_train, y_train)
			my_prediction = classifier2.predict(list) 
			print("RandomForest")


		print('')
		print(my_prediction)
		print('')

		if(my_prediction == 1):
			my_prediction = teamDict[TeamA]
		elif(my_prediction == -1):
			my_prediction = teamDict[TeamB]
		else:
			my_prediction = "Draw"

	return render_template('result1.html', prediction = my_prediction,li =finalList)




#-----------------------------------------------



teamdictcricket = {
		  1 : "Kolkata",
		  2 : "Bangalore",
		  3 : "Chennai",
		  4 : "Punjab",
		  5 : "Rajasthan",
		  6 : "Delhi",
		  7 : "Mumbai",
		  9 : "Hyderabad",
		  10 : "Pune",
		  11 : "Hyderabad",
		  12 : "Gujarat",
		  
		}


@app.route('/ckt_dashboard')
def ckt_dashboard():
	return render_template('ckt_dashboard.html')
	

@app.route('/ckt_predict')
def ckt_predict():
	return render_template('index1.html')

@app.route('/ckt_rankings')
def ckt_rankings():
	return render_template('ckt_rankings.html')


@app.route('/ckt_videos')
def ckt_videos():
	return render_template('ckt_videos.html')

@app.route('/ckt_schedule')
def ckt_schedule():
	return render_template('ckt_schedule.html')

@app.route('/ckt_news')
def ckt_news():
	return render_template('ckt_news.html')


@app.route('/ckt_output', methods=['POST'])
def ckt_output():
	
	

	df = pd.read_csv("data/Sheet1.csv")
	X = np.array(df.drop(['Result'],axis = 1))
	y = np.array(df['Result'])

	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)

	if request.method == 'POST' :
		
		TeamA = int( request.form['TeamA'])
		
		TeamB = int(request.form['TeamB'])
		condt =int( request.form['condt'])
		tosswin = int( request.form['tosswin'])
		has =float( request.form['has'])
		hds = float(request.form['hds'])
		aas =float( request.form['aas'])
		ads = float(request.form['ads'])
		tosswin = int( request.form['tosswin'])
		Algorithm = int( request.form['Algorithm'])



		list = [[TeamA,TeamB,condt,tosswin,has,hds,aas,ads]]
		finallistcricket = [[teamdictcricket[TeamA],teamdictcricket[TeamB],has,hds,aas,ads]]
		print('')
		print(list)
		print('')

		#data=[[40,	1,	18,	1,	0,	0,	0,	1,	0.25	,0,	0,	1,	2,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	1,	2,	2,	0,	0,	0,	0,	1,	1,	1]]		
		if(Algorithm==1):
			from sklearn.neighbors import KNeighborsClassifier
			classifier1 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
			classifier1.fit(X_train, y_train)
			my_prediction = classifier1.predict(list) 
			print("KNearestNeighbour")
		else:
			from sklearn.ensemble import RandomForestClassifier  
			classifier = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
			classifier.fit(X_train,y_train)
			my_prediction = classifier.predict(list) 
		    

		print('')
		print(my_prediction)
		print('')

		if(my_prediction == 1):
			my_prediction = teamdictcricket[1]
		elif(my_prediction == 2):
			my_prediction = teamdictcricket[2]
		elif(my_prediction == 3):
			my_prediction = teamdictcricket[3]	
		elif(my_prediction == 4):
			my_prediction = teamdictcricket[4]
		elif(my_prediction == 5):
			my_prediction = teamdictcricket[5]
		elif(my_prediction == 6):
			my_prediction = teamdictcricket[6]
		elif(my_prediction == 7):
			my_prediction = teamdictcricket[7]
		elif(my_prediction == 9):
			my_prediction = teamdictcricket[9]
		elif(my_prediction == 10):
			my_prediction = teamdictcricket[10]
		elif(my_prediction == 11):
			my_prediction = teamdictcricket[11]
		elif(my_prediction == 12):
			my_prediction = teamdictcricket[12]
		else:
			my_prediction = "Draw"

	return render_template('result1.html', prediction = my_prediction,li = finallistcricket)


if __name__ == '__main__':
	app.run(debug=True)