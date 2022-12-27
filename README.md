# Predicting Stock Prices Using Machine Learning

This repository contains a simple AI system aimed at predicting the trend of the **NVIDIA stock** prices using machine learning.
The data needed for the analysis are imported through `yfinance`, an open-source tool that uses **Yahoo**'s publicly available APIs to retrieve data about listed stocks.

The financial and banking sectors are incredibly **data-rich**, with millions of transactions and transfers occurring every day.
Machine learning models cab be used to understand emerging and underlying trends in order to gain advantage in the **financial sector**.

Check out the [Colab Notebook](stock_prediction.ipynb) to review the execution of the project. 

## Requirements

I should write about **software dependencies** but I'm not that sure about them so TBD

## About the Model

The goal of the model is to predict whether we should **buy** the stock (**target = 1**), meaning that the price is going to increase in the future, or **sell** the stock (**target = 0**), when the price is going to decrease in the future.
The model is trained using data about NVIDIA stock (*Open, High, Low, Close, Volume*) analyzed over a given period, as well as data about main business stakeholders and key financial indicators. 

The **regression** and **classification** algorithms used in the project are featured by the `scikit-learn` machine learning library, while data modeling is mainly performed using the `pandas` software library. 

[![Renovate enabled](https://img.shields.io/badge/renovate-enabled-brightgreen.svg)](https://renovatebot.com/)
[![Build status](https://github.com/renovatebot/renovate/workflows/build/badge.svg)](https://github.com/renovatebot/renovate/actions)




