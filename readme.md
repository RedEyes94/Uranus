[![License](https://img.shields.io/github/license/italia/bootstrap-italia.svg)](https://www.apache.org/licenses/)

# URANUS: Design and Implementation of a Radio Frequency Tracking, Classification and Identification Machine Learning System of Unmanned Aircraft Vehicles

This repository contains an open source implementation of the NATO-published challenge solution related 
to UAS drone detection and tracking available at the link https://www.kaggle.com/c/icmcis-drone-tracking/overview
The task we set with this project is to detect, track, and identify Class 1 UAS within a protected area, 
taking advantage of detections made by various sensors. Drones will be tracked with respect 
to position, speed, and altitude. The observations, in the case of drone detection, will then be classified by drone type 
(fixed or rotating wings) and the name of the potential detected UAS will be indicated 
(e.g. DJI Mavic Pro).

![image](Utility/ML-Design.jpg)

# Prerequisites


This repository assumes Python 3.8.3 or newer is used.

> Project dependencies



| Library  | README |
| ------ | ------ |
| imblearn | https://imbalanced-learn.org/stable/ |
| scikit-learn 1.0.2| https://scikit-learn.org/ |
| NumPy 1.22.3| https://numpy.org/ |
| Pandas 1.4.2 | https://pandas.pydata.org/ |
| pickle 4.0 | https://docs.python.org/3/library/pickle.html|
| Matplotlib | https://matplotlib.org/ |
| Scipy 1.8.0|https://scipy.org/|



## Istructions

Run the following command to get the project
```js
git clone https://github.com/RedEyes94/Uranus.git
```

Open command prompt, and go to project path

```js
cd /path/to/Uranus
```
Run the following commands
```js
$ py console.py
```
After the script execution a menu will be shown. In this menu you can choose one of the available scenarios

```js
*************************************************
*****************  Uranus  **********************
*************************************************

 All available test scenarios

1 )  1a : only one drone in flight
2 )  1b : only one drone in flight
3 )  2a : two drones flying
4 )  2b : two drones flying
5 )  2c : two drones flying
6 )  2d : two drones flying
7 )  3a : only one drone in flight (Parrot)
```
After the scenario has been selected, the time intervals of the flight will be displayed. Enter the values within the range.

```js
*************************************************
*****************  URANUS  **********************
*************************************************

 All available test scenarios

1 )  1a : only one drone in flight
2 )  1b : only one drone in flight
3 )  2a : two drones flying
4 )  2b : two drones flying
5 )  2c : two drones flying
6 )  2d : two drones flying
7 )  3a : only one drone in flight (Parrot)

Choose test scenario:1a
Choose range datatime into : 2020-09-29 13:00:09 and 2020-09-29 13:09:16
Insert first range value:  2020-09-29 13:04:09
Insert second range value:  2020-09-29 13:09:16
```
After setting the date range, the software will display in order: the list of detections in the indicated range, the projection of latitudes and longitudes on a map showing the flight path of the drone, for each type of drone the number of relative detections and finally it will display an image containing the drone(s) detected in the specified range.

![image](Utility/table_screen.PNG)
![image](Utility/map_screen.PNG)
![image](Utility/numeri_droni.PNG)
![image](Utility/drone.PNG)

## Repository Structure

```js
├── 1-AlviraClusteringResults            # Contains the files obtained from the data clustering phase on Alvira
├── 2-AnovaResults                       # Contains the files obtained from the Anova analysis
├── 3-DataAnalysisResults                # Contains the files obtained from the data Analysis (heatmap, histogram, missing rata)
├── 4-DataCleanResults                   # Contains the files obtained from the data cleaning 
├── 5-MapResults                         # Contains the map results of real drone flight and prediction of drone flight in test/training scenarios
├── 6-MergeResults                       # Contains the files obtained from the merge of Alvira, Arcus, Diana, Venus e Drone datasets
├── 7-Models                             # Contains the extracted models 
├── 8-OutliersResults                    # Contains the files obtained from the outliers analysis 
├── 9-PredictionResults                  # Contains the files obtained from the prediction for training and test scenarios 
├── 10-Scenari                           # Contains the raw files related to each scenario (train/test) with sensors and drone data 
├── 11-Sensors                           # Contains the classes of sensors 
├── 12-Steps                             # Contains all the step followed by Uranus
│   ├── Step_0                           # Contains py files for Outliers detection, Data analisys and Clean
│   ├── Step_1                           # Contains py files for clustering 
│   ├── Step_2                           # Contains py files for merge all sensors files
│   ├── Step_3                           # Contains py files for anova analisys, creation of learning models  
│   ├── Step_4                           # Contains py files for testing models generated in Step 3  
├── SubmissionFileTestResults            # Contains csv files genereted after prediction in Step 4  
├── Utility                              # Contains py files with some useful functions (e.g. GPS)
```
## Authors
<ul>
<li>Giuseppe De Marzo</li>
(Indipendent Researcher Politecnico di Bari, Bari (Italy), gdemarzo94 at gmail dot com)
<li>Domenico Lofù</li>
(Dept. of Electrical and Information Engineering (DEI), Politecnico di Bari, Bari (Italy), domenico dot lofu at poliba dot it\br>
and Innovation, Marketing & Technology, Exprivia S.p.A., Molfetta (Italy), domenico dot lofu at exprivia dot com)
<li>Pietro Tedeschi</li>
(Ph.D, Senior Security Researcher @ Technology Innovation Institute, Secure Systems Research Center, Abu Dhabi, pietro dot tedeschi at tii dot ae)
</ul>

## License
The code is provided under the Apache License. If you need it under a more permissive 
license then please contact me.
