# As the model is only accurate after about 20 samples, and due to the high R2 score there is a possiblity of over fitting I will also run a linear regression
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, learning_curve
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
# using joblib as it is more secure and will be predicting data based on GDPR and as all within sklearn the compatability isn't an issue
from joblib import dump

def linear_regression(X, y, X_train, X_test, y_train, y_test, type_science, encoder, file_name):
    lr = LinearRegression()

    # fit the model to the training data
    lr.fit(X_train, y_train)

    y_pred_unrounded = lr.predict(X_test)

    if type_science.lower()[0] == 'c':
        # Rounding grades back to their original catagorical representations
        y_pred = np.vectorize(lambda x: round(x * 2) / 2)(y_pred_unrounded)
    elif type_science.lower()[0] == 't':
        y_pred = np.vectorize(lambda x: round(x))(y_pred_unrounded)
    else:
        print("Incorrect type")

    # evaluate the model performance using mean squred error, root and r squared
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)


    y_train_size = y_pred.shape[0]
    print("Mean Squared Error (MSE): {:.2f}".format(mse))
    print('Train MSE {:.2f}'.format(mean_squared_error(y_train[:y_train_size], y_pred)))
    print('testMSE {:.2f}'.format(mean_squared_error(y_test, y_pred)))
    print("Root Mean Squared Error (RMSE): {:.2f}".format(rmse))
    print("R2 score:", r2)

    # this again has an extremely high R2 score, so may be over fit

    # create a kfold to check scores are similar
    kfold = KFold(n_splits=10, shuffle=True, random_state=15)
    scores = cross_val_score(lr, X, y, cv=kfold, scoring='r2')
    # the cross-validated scores are very similar, reducing the chance that the model is overfitted
    print('Cross-validation scores:', scores)

    # create a learning curve to see if data is overfitting
    train_sizes, train_scores, test_scores = learning_curve(lr, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
                                                            
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
                                                            
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training score')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='test score')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.xlabel('Number of training samples')
    plt.ylabel('Score')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig("model_graphs/" + file_name + "_linear.png", )

    # if data is good give option to save and name to save
    save = input('should it be saved? ')
    if save[0].strip().lower() == 'y':
        if type_science.lower()[0] == 'c':
            dump(lr, 'combined_models/combined_linear.joblib')
            dump(encoder, 'combined_models/combined_linear_encoding.joblib')
            print('Model save')
        else:
            dump(lr, 'triple_models/triple_linear.joblib')
            dump(encoder, 'triple_models/triple_linear_encoding.joblib')
            print('Model save')

