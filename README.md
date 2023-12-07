# ***Time-series Modelling Case Studies for Huawei Internship***

The purpose of the second round of interview is to assess your problem solving skills in real-world data science problems in the areas of time-series anomaly detection, prediction, and clustering. 

Data are contained in two subfolders: 

  1.	AD&P (data for anomaly detection and prediction case studies) 
  2.	C (data for clustering case study)



## **Case study 1: Anomaly Detection**
The task is to produce time-series point-anomaly detection models for each of the 25 Key Performance Indicators (KPI) time-series contained in AD&P folder. 
- In detail: 
  -	Each row associates a “time-stamp” with a “kpi_value”. The anomaly detector should be trained to detect point-anomalies in “kpi_value”. 

  -	For the majority of time-series the problem is unsupervised (no point-anomaly labels). For some KPIs (datasets 201-206) we have also access to labelled data. In those cases labels can be taken into account if needed (optional). 

  -	Each of the 25 time-series needs to be modelled independently of others.
  
- Deliverables: 
  -	Slides with description of the anomaly detection method chosen. 
  -	Slides with visualisation of anomaly detection output in whole time-series, for each of the 25 time-series. Example: 


-	During presentation, issues of scalability, adaptation, model selection, generalisation will be discussed.



## Case 2: Prediction

The task is to produce time-series prediction models for each of the 25 Key Performance Indicators (KPI) time-series contained in AD&P folder. 
- In detail: 
  -	For each of the 25 time-series train a model to forecast the kpi_value at time t+1, t+2, t+3, t+4, t+5, given information up to time t.


- Deliverables: 
  -	Slides with description of the prediction method chosen. 
  -	Slides with train/test prediction performance assessment. 
  -	Slides with visualisation of prediction versus actual time-series values. Example: 
  
  
  -	During presentation, issues of scalability, adaptation, model selection, generalisation will be discussed.


## **Case study 3: Clustering The task is to cluster the 23 time-series contained in the C folder.**

- In detail: 
  -	Monthly time-series data: “value” column associated with “date” column. 
  -	Number of optimal cluster should be emergent through your analysis.

- Deliverables: 
  -	Slides with description of the clustering method chosen. 
  -	Visualisation of clustering results. 
  -	During presentation, issues of scalability and model selection will be discussed. 
  
  General note: During presentation, the candidate may be asked to go through his/her Python implementation of any of the case studies – coding style will be assessed.