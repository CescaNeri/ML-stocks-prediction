# Predicting Stock Prices Using Machine Learning

This repository contains a simple AI system aimed at predicting the trend of the **NVIDIA stock** prices using machine learning.
The data needed for the analysis are imported through `yfinance`, an open-source tool that uses Yahoo's publicly available APIs to retrieve data about listed stocks.

The goal of the model is to predict whether we should **buy** the stock (`target = 1`), meaning that the price is going to increase in the future, or **sell** the stock (`target = 0`), when the price is going to decrease in the future.
The model is trained using data about the NVIDIA stock (Open, High, Low, Close, Volume) for a given period, as well as data about main business stakeholders and key financial indicators. 

[![Renovate enabled](https://img.shields.io/badge/renovate-enabled-brightgreen.svg)](https://renovatebot.com/)
[![Build status](https://github.com/renovatebot/renovate/workflows/build/badge.svg)](https://github.com/renovatebot/renovate/actions)




