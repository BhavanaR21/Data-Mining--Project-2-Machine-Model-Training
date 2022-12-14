# Data-Mining--Project-3-Machine-Model-Training

Project 2: Machine Model Training

Purpose

In this project you will use a training dataset to train and test a machine model. The purpose is to distinguish between meal and no meal time series data.

Objectives

● Develop code to train a machine model.
● Assess accuracy of machine model.

Project Description

In this project you will train a machine model to assess whether a person has eaten a meal or not eaten a 
meal.  A training data set is provided.

Directions

Meal data can be extracted as follows:
From the InsulinData.csv file, search the column Y for a non NAN non zero value. This time indicates 
the start of meal consumption time tm. Meal data comprises a 2hr 30 min stretch of CGM data that 
starts from tm-30min and extends to tm+2hrs.
No meal data comprises 2 hrs of raw data that does not have meal intake.

Extraction: Meal data

Start of a meal can be obtained from InsulinData.csv. Search column Y for a non NAN non zero value. 
This time indicates the start of a meal. There can be three conditions:
● There is no meal from time tm to time tm+2hrs. Then use this stretch as meal data
● There is a meal at some time tp in between tp>tm and tp< tm+2hrs. Ignore the meal data at 
time tm and consider the meal at time tp instead
● There is a meal at time tm+2hrs, then consider the stretch from tm+1hr 30min to tm+4hrs as 
meal data.

Extraction: No Meal data

Start of no meal is at time tm+2hrs where tm is the start of some meal. We need to obtain a 2 hr stretch 
of no meal time. So you need to find all 2 hr stretches in a day that have no meal and do not fall within 
2 hrs of the start of a meal.
Handling missing data:
You have to carefully handle missing data. This is an important data mining step that is required for 
many applications. Here there are several approaches: a) ignore the meal or no meal data stretch if the 
number of missing data points in that stretch is greater than a certain threshold, b) use linear interpolation (not 
a good idea for meal data but maybe for no meal data), c) use polynomial regression to fill up missing data 
(untested in this domain). Choose wisely.
Feature Extraction and Selection:
You have to carefully select features from the meal time series that are discriminatory between meal 
and no meal classes.

Test Data:

The test data will be a matrix of size N×24, where N is the total number of tests and 24 is the size of 
the CGM time series. N will have some distribution of meal and no meal data.
Note here that for meal data you are asked to obtain a 2 hr 30 min time series data, while for no meal 
you are taking 2 hr. However, a machine will not take data with different lengths. Hence, in the feature 
extraction step, you have to ensure that features extracted from both meal and no meal data have the 
same length.
