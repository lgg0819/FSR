# -*- coding: utf-8 -*-
from ROOT import *
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import tensorflow as tf
from keras import Sequential, optimizers
from keras.layers import Dense, Activation


def main():

	target = 'target'
	features = ['higgsness','topness']

	df_hh = pd.read_csv("htness_sig.csv")
	df_tt = pd.read_csv("htness_bkg.csv")

	data = pd.concat([df_hh, df_tt], ignore_index=True)

	train,valid = train_test_split(data, test_size=0.3, random_state=52)

	X_train = train[features]
	y_train = train[target]
	X_valid = valid[features]
	y_valid = valid[target]

	scaler = StandardScaler()

	X_train = scaler.fit_transform(X_train)
	X_valid = scaler.fit_transform(X_valid)

#	logistic = LogisticRegression()
#	logistic.fit(X_train, y_train)
#	score_logistic = logistic.score(X_train, y_train)

	model = Sequential()

	model.add(Dense(1, input_dim=2, activation='sigmoid',
		kernel_initializer='glorot_normal',
		bias_initializer='zeros'))

	model.compile(loss='binary_crossentropy', optimizer='adam')

	model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)

	y_pred = model.predict(X_valid)
	y_real = y_valid.values.tolist()

	# ======================== Visualization Part ========================

	c1 = TCanvas("c1","Logistic Regression Model Prediction_DL", 900,600)

	prev_hh = TH1D("h1","Logistic Regression Prediction Score_DL", 40,0,1.0)
	prev_tt = TH1D("h1","Logistic Regression Prediction Score_DL", 40,0,1.0)

	for i in range(len(y_real)):
		if y_real[i] == 1:
			prev_hh.Fill(y_pred.item(i))
		if y_real[i] == 0:
			prev_tt.Fill(y_pred.item(i))

	prev_hh.Scale(1/prev_hh.GetEntries())
	prev_tt.Scale(1/prev_tt.GetEntries())

	hist_hh = TH1D("h1","Logistic Regression Prediction Score_DL", 40,0,1.0)
        hist_tt = TH1D("h1","Logistic Regression Prediction Score_DL", 40,0,1.0)

	for i in range(1,hist_hh.GetNbinsX()+1):
		hist_hh.SetBinContent(i,prev_hh.GetBinContent(i))
		hist_tt.SetBinContent(i,prev_tt.GetBinContent(i))

	hist_hh.SetLineColor(2)
	hist_hh.SetFillColor(2)
	hist_hh.SetFillStyle(3004)
	hist_tt.SetFillColor(4)
	hist_tt.SetFillStyle(3005)

	hist_tt.GetXaxis().SetTitle("Prediction Score")
	hist_tt.GetYaxis().SetTitle("Entry")

	c1.cd()
	hist_tt.Draw()
	hist_hh.Draw("same")

	pred_ld = y_pred.flatten()
	y_pred = np.where(pred_ld > 0.6, 1, 0)

	print(accuracy_score(y_real, y_pred))

	input("Press Enter to continue...")


if __name__ == "__main__":
        main() 
