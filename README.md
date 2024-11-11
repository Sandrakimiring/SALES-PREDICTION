# SALES-PREDICTION

# Sales Prediction Using Simple Linear Regression

## Overview

This project focuses on predicting future sales based on advertising expenditure using a Simple Linear Regression model. The goal is to understand how different amounts spent on advertising (in various platforms) influence the amount of sales generated. This is a classic use case for machine learning, particularly in business analytics, where predicting sales accurately can optimize advertising budgets and maximize revenue.

## Dataset

The dataset used in this project contains historical data on advertising expenditures across different platforms (TV, Radio, Newspaper) and corresponding sales figures. This data is leveraged to build a predictive model that estimates sales based on a given advertising budget.

## Project Structure

- `sales_prediction.ipynb` - The Jupyter notebook containing all the steps of data exploration, preprocessing, model building, evaluation, and predictions using Simple Linear Regression.
- `data/` - Directory containing the dataset (in `.csv` format).
- `requirements.txt` - List of dependencies required to run the project.

## Requirements

This project requires the following Python libraries:

- `pandas` - For data manipulation.
- `numpy` - For numerical operations.
- `matplotlib` - For plotting visualizations.
- `seaborn` - For enhanced data visualization.
- `scikit-learn` - For implementing the Linear Regression model and evaluating its performance.

You can install the necessary dependencies using the following command:

```
pip install -r requirements.txt
```

## Steps

### 1. Load and Explore the Data
The project begins by loading the sales data into a pandas DataFrame and performing initial exploratory data analysis (EDA). This includes checking for missing values, understanding the structure of the dataset, and visualizing the relationship between advertising expenditure and sales.

### 2. Data Preprocessing
The dataset undergoes preprocessing, where irrelevant columns are dropped, and the data is cleaned and prepared for model training. The target variable (Sales) is separated from the features (TV, Radio, and Newspaper ad spend).

### 3. Train-Test Split
The data is split into training and testing sets, ensuring that the model is evaluated on unseen data after training.

### 4. Model Building
A Simple Linear Regression model is used to fit the training data. The model attempts to find the best linear relationship between advertising expenditure and sales.

### 5. Model Evaluation
The model's performance is evaluated using metrics such as:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R-squared** to measure how well the model fits the data.

### 6. Predictions
Once the model is trained and evaluated, it is used to make sales predictions based on new advertising budgets. This is the final step, where the model is applied to predict sales for given advertising spends.

## Example Usage

Once the notebook has been run and the model is trained, you can make predictions on new data as follows:

```python
from sklearn.linear_model import LinearRegression

# Example advertising spend
new_data = [[150]]  # Example: $150 TV advertising spend

# Train model on your data (refer to the notebook for training details)
model = LinearRegression()
model.fit(X_train, y_train)

# Predict sales
predicted_sales = model.predict(new_data)
print(f"Predicted Sales: {predicted_sales[0]}")
```

This will give the predicted sales based on a new advertising expenditure input.

## Conclusion

By using simple linear regression, we are able to create a model that forecasts sales based on the amount spent on advertising. The key takeaway is the importance of data in making informed decisions about budget allocation, and how machine learning can be applied to make these predictions more accurate.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Kaggle](https://www.kaggle.com/) for providing the dataset.
- [Scikit-learn](https://scikit-learn.org/) for machine learning algorithms and tools.
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for visualization libraries.

---

