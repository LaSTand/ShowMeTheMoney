from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# Read lotto statistics
lotto_stats = pd.read_csv('lottery.csv')

X_array = [0, 0,
	lotto_stats[['1st']],
	lotto_stats[['1st', '2nd']],
	lotto_stats[['1st', '2nd', '3rd']],
	lotto_stats[['1st', '2nd', '3rd', '4th']],
	lotto_stats[['1st', '2nd', '3rd', '4th', '5th']],
	lotto_stats[['1st', '2nd', '3rd', '4th', '5th', '6th']]]

Y_array = [0, 0, 
	lotto_stats[['2nd']], 
	lotto_stats[['3rd']], 
	lotto_stats[['4th']], 
	lotto_stats[['5th']], 
	lotto_stats[['6th']], 
	lotto_stats[['bonus']]]

def linear_regression_predict(indep, dep):
	linear_regression = linear_model.LinearRegression()
	linear_regression.fit(X=pd.DataFrame(indep), y=dep)
	return linear_regression.predict(X=pd.DataFrame([init_array]))

def append_new_value(array, value):
	return np.append(array, value)

def main():
	global init_array
	
	for i in range(1, 31):
		init_array = np.array([i])	# The first value is 1~30
		for i in range(2, 8):
			temp_prediction = linear_regression_predict(X_array[i], Y_array[i])
			init_array = append_new_value(init_array, temp_prediction)
		print(np.rint(init_array))

if __name__ == '__main__':
	main()