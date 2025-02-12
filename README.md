Practical Machine Learning Project
==================================

This is a project for the peer assessment activity in the __Practical Machine Learning__ course provided by Johns Hopkins University through Coursera. 
The purpose is to apply machine learning techniques to model predictions and compare the results from different approaches. The document with the analysis CourseProject.md is available at the present repository and also through the following link: 

https://github.com/jldengra/PML_Project/blob/master/CourseProject.md

The following files are included in this repository:

* __README.md__ and __README.html__

	Overall description of the project repository and files. 
	
* __CourseProject.Rmd__, __CourseProject.Rmd__ and __CourseProject.html__

	R markdown and compiled HTML files describing the analysis.
       
* __data/pml-testing.csv__ and __data/pml-training.csv__

	Data files for test and training.

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Data

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment. 

## Assignment

The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. We may use any of the other variables to predict with. We should create a report describing how we built your model, how we used cross validation, what we think the expected out of sample error is, and why we made the choices you did. We will also use your prediction model to predict 20 different test cases. 