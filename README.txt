🏏 ODI Cricket Strike Rate Classifier
This project implements a Machine Learning model to classify the expected Strike Rate (SR) category of a cricket player's innings based on fundamental performance metrics. It uses a Random Forest Classifier optimized to handle imbalanced cricket data using the SMOTE oversampling technique.
 Model Objective
The goal of the model is to categorize an innings into one of three strike rate performance tiers:
* Low (0): SR < 70
* Medium (1): 70 <= SR < 120
* High (2): SR >= 120
 Key Results
The final tuned model achieved strong performance, especially in differentiating the minority (High and Medium SR) classes, which is crucial for real-world analysis.
* Accuracy: ~86.5%
* Primary Improvement: Feature Engineering (adding run_rate_per_ball) significantly improved F1-scores for minority classes.
 Requirements
To run this script, you need Python 3 and the following libraries:
Library
	Installation Command
	Purpose
	pandas
	pip install pandas
	Data manipulation
	numpy
	pip install numpy
	Numerical operations
	scikit-learn
	pip install scikit-learn
	ML model and scaling
	imbalanced-learn
	pip install imbalanced-learn
	SMOTE for handling imbalanced data
	matplotlib
	pip install matplotlib
	Plotting (Confusion Matrix, Feature Importance)
	seaborn
	pip install seaborn
	Enhanced data visualization
	Data File
This project requires a data file named:
* ODI Cricket Data new.csv
This file must be placed in the same directory as the cricket_model.py script. It should contain relevant columns, including: total_runs, total_balls_faced, and strike_rate.
How to Run the Script
1. Clone the Repository:
git clone https://github.com/EssaBinTahir-08/cricket-sr-classifier.git
cd cricket-sr-classifier

2. Install Dependencies:
pip install -r requirements.txt 
# (assuming you create a requirements.txt with the libraries listed above)

3. Place Data: Ensure the ODI Cricket Data new.csv file is in the project root.
4. Execute the Model:
python cricket_model.py

The script will output the class distribution, model evaluation metrics (Accuracy, Classification Report), and display plots for the Confusion Matrix and Feature Importance.
Model Methodology Highlights
   1. Feature Engineering: The critical feature run_rate_per_ball (total_runs / total_balls_faced) was added to provide the tree-based model with a clear, direct ratio for better classification.
   2. SMOTE Oversampling: Used to synthesize data for the minority classes (Medium and High SR), preventing the model from becoming biased towards the dominant Low SR class.
   3. StandardScaler: Features were scaled to normalize the input data, although Random Forest is relatively robust to scaling.
   4. Optimized Hyperparameters: The Random Forest was configured using parameters (n_estimators=50, max_depth=15, etc.) found via cross-validation.