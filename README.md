# Graduate Starting Salary Predictor
Using regression modules to predict the starting salaries of graduates

## Goal

The aim of this project is to build a regression model that predicts a graduate’s starting salary. The prediction is based on key factors derived from their education, skills, and extracurricular activities (e.g. GPA, university ranking, soft skills, internships). This was attempted using five regression methods and utilizing the best performing model.

## Data used

- **Source:** [Education and Career Success Dataset](https://www.kaggle.com/datasets/adilshamim8/education-and-career-success/data)
- **Overview:**  
  The dataset contains 5,000 records capturing students’ academic backgrounds, skills, and early career outcomes. Key features include:
  - **Academic Metrics:** High School GPA, University GPA, SAT Score, University Ranking
  - **Extracurricular Experiences:** Internships, Projects, Certifications, Soft Skills, Networking Scores
  - **Career Outcomes:** Starting Salary, Job Offers, Career Satisfaction
 
<p float="left">
  <img src="README_figures/data.png" width="340" />
</p>

 
## Model Selection

Several regression models were implemented and compared to predict graduate starting salaries. Here’s a more detailed breakdown of each:

### 1. Feedforward Neural Network
- **Framework:** Implemented using PyTorch.
- **Architecture:**  
  - Two hidden layers with ReLU activation functions.
  - The network architecture was designed to capture non-linear relationships in the data.
- **Training Strategy:**  
  - Data was split into 70% for training, 20% for testing, and 10% for validation.
  - Mean Squared Error (MSE) was used as the loss function, and the model’s performance was tracked by monitoring RMSE over multiple epochs.
- **Objective:**  
  - To leverage deep learning for capturing complex patterns that traditional linear models might miss.

### 2. Linear Regression
- **Purpose:** Served as a baseline model to set a reference performance.
- **Method:**  
  - A simple linear model that assumes a linear relationship between the predictors (e.g., GPA, internships) and the starting salary.
- **Evaluation Metrics:**  
  - Evaluated using RMSE, Mean Absolute Error (MAE), and the R² score on both test and validation sets.
- **Strengths & Limitations:**  
  - Quick to train and interpret, but may underfit if the relationships in the data are non-linear.

### 3. Ridge and Lasso Regression
- **Approach:**  
  - Both are extensions of linear regression that include regularization terms.
- **Ridge Regression:**  
  - Adds an L2 penalty to the loss function, which helps to shrink the coefficients and reduce overfitting.
- **Lasso Regression:**  
  - Incorporates an L1 penalty, which can drive some coefficients to zero, effectively performing feature selection.
- **Insights:**  
  - Coefficient analysis revealed that features such as internships and soft skills have strong predictive power for starting salaries.
- **Evaluation:**  
  - Performance was compared via RMSE on the validation set, and both methods provided similar error rates, with Ridge slightly outperforming Lasso.

### 4. OLS Regression
- **Method:**  
  - Ordinary Least Squares regression was implemented using Statsmodels.
- **Enhancements:**  
  - Interaction terms were introduced (e.g., combining top university ranking with fields of study like business or law) to capture synergistic effects.
- **Evaluation:**  
  - Performance was measured using adjusted R² to account for the number of predictors, alongside RMSE.
- **Interpretation:**  
  - The OLS model provided a detailed statistical summary, including p-values and confidence intervals for each predictor, which helped in understanding the significance of the factors involved.

### 5. Model Comparison
- **Comparative Analysis:**  
  - All models were evaluated on the same validation set with RMSE as the primary metric.
    
## Results and Conclusions

### Results

### Conclusions

## Technologies Used

- **Python 3.x**
- **PyTorch** (for the neural network model)
- **scikit-learn** (for Linear, Ridge, Lasso, and OLS regression)
- **Pandas & NumPy** (for data manipulation)
- **Matplotlib & Seaborn** (for data visualization)
- **Statsmodels** (for OLS regression analysis)
