import pandas as pd
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Regression:

    def __init__(self, X_train, X_test, y_train, y_test, df):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.df = df

    def run_regression(self):

        print("ridge")
        regressor = self.train_ridge(self.X_train, self.X_test, self.y_train, self.y_test)
        predicted = self.test_ridge(self.X_train, self.X_test, self.y_train, self.y_test, regressor)
        self.linear_assumption(predicted)

    def linear_assumption(self, predictions):

        df_results = pd.DataFrame({'Actual': self.y_test, 'Predicted': predictions})
        df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])
        """
        Linearity: Assumes there is a linear relationship between the predictors and
                   the response variable. If not, either a polynomial term or another
                   algorithm should be used.
        """
        print('\n=======================================================================================')
        print('Assumption 1: Linear Relationship between the Target and the Features')

        print('Checking with a scatter plot of actual vs. predicted. Predictions should follow the diagonal line.')

        # Plotting the actual vs predicted values
        sns.lmplot(x='Actual', y='Predicted', data=df_results, fit_reg=False, height=7)

        # Plotting the diagonal line
        line_coords = np.arange(df_results.min().min(), df_results.max().max())
        plt.plot(line_coords, line_coords,  # X and y points
                 color='darkorange', linestyle='--')
        plt.title('Actual vs. Predicted')
        plt.show()
        print('If non-linearity is apparent, consider adding a polynomial term')

    def train_ridge(self, X_train, X_test, y_train, y_test):
        # Train the ridge regression model lambda = 0.1
        rr = Ridge(alpha=1)

        rr.fit(self.X_train, self.y_train)

        print("bias is " + str(rr.intercept_))
        print("coefficients  are " + str(rr.coef_))

        y_train_pred = rr.predict(self.X_train)

        mae = mean_absolute_error(y_train_pred, self.y_train)
        mse = mean_squared_error(y_train_pred, self.y_train)
        rmse = np.sqrt(mse)

        print('prediction for training set:')
        print('MAE is: {}'.format(mae))
        print('MSE is: {}'.format(mse))
        print('RMSE is: {}'.format(rmse))
        print('R-Squared is: {}'.format(rr.score(self.X_train, self.y_train)))
        return rr

    def test_ridge(self, X_train, X_test, y_train, y_test, rr):
        y_test_pred = rr.predict(self.X_test)
        predicted = y_test_pred
        mae = mean_absolute_error(y_test_pred, self.y_test)
        mse = mean_squared_error(y_test_pred, self.y_test)
        rmse = np.sqrt(mse)

        print('prediction for testing set: Ridge')
        print('MAE is: {}'.format(mae))
        print('MSE is: {}'.format(mse))
        print('RMSE is: {}'.format(rmse))
        print('R-Squared is: {}'.format(rr.score(self.X_test, self.y_test)))

        # Visualize the Ridge Regression model
        labels = ['House1', 'House2', 'House3', 'House4', 'House5']
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, y_test[0:5], width, label='ground truth')
        rects2 = ax.bar(x + width / 2, y_test_pred[0:5], width, label='prediction')

        ax.set_ylabel('Price')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        plt.show()

        self.show_coeff(rr)
        return predicted

    def show_coeff(self, rr):
        headers = list(self.df.columns.values)
        headers.remove('price')
        res = pd.DataFrame(rr.coef_,
                           headers,
                           columns=['coef']).sort_values(by='coef', ascending=False)
        print(res)




