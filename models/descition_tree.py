import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_score, learning_curve
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from joblib import dump


def descition_tree(X, y, X_train, X_test, y_train, y_test, type_science, encoder, file_name):
    if type_science == 'c':
        dtr = DecisionTreeRegressor(random_state=142, max_depth=None, min_samples_leaf=1, min_samples_split=5)
    elif type_science == 't':
        dtr = DecisionTreeRegressor(random_state=142, max_depth=3, min_samples_leaf=2, min_samples_split=10)
    else:
        print('No model loaded')

    dtr.fit(X_train, y_train)

    y_pred_unrounded = dtr.predict(X_test)

    y_pred = np.vectorize(lambda x: round(x * 2) / 2)(y_pred_unrounded)

    y_train_size = y_pred.shape[0]
    mse = mean_squared_error(y_test, y_pred)
    print('Train MSE {:.2f}'.format(mean_squared_error(y_train[:y_train_size], y_pred)))
    print('test MSE {:.2f}'.format(mean_squared_error(y_test, y_pred)))
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print('Mean Squared Error (MSE):', mse)
    print('Root Mean Squared Error (RMSE):', rmse)
    print('R2 Score:', r2)

    # to check for overfitting

    kfold = KFold(n_splits=10, shuffle=True, random_state=15)
    scores = cross_val_score(dtr, X, y, cv=kfold, scoring='r2')
    # the cross-validated scores are very similar, reducing the chance that the model is overfitted
    print('Cross-validation scores:', scores)

    train_sizes, train_scores, test_scores = learning_curve(dtr, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training score')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='test score')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.xlabel('Number of training samples')
    plt.xscale('log')
    plt.ylabel('Score')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig("model_graphs/" + file_name + "_descition.png", )
    plt.clf()

    save = input('should it be saved? ')
    if save[0].strip().lower() == 'y':
        if type_science == 'c':
            dump(dtr, 'combined_models/combined_descition_tree.joblib')
            dump(encoder, 'combined_models/combined_descition_tree_encoding.joblib')
            print('Model save')
        else:
            dump(dtr, 'triple_models/triple_descition_tree.joblib')
            dump(encoder, 'triple_models/triple_descition_tree_encoding.joblib')
            print('Model save')
