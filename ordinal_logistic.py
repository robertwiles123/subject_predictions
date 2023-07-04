import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from grades_packages import df_columns

combined = pd.read_csv('csv_clean/clean_combined.csv')

X = combined[df_columns.combined_dependent()]
y = combined[df_columns.combined_independent()[0]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=64)

model = LogisticRegression(multi_class='multinomial', solver='newton-cg')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

                                                       
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
plt.ylim([0.99, 1.01])
plt.show()
plt.savefig("model_graphs/combined_ordinal_logistic.png", )


save = input('should it be saved? ')
if save[0].strip().lower() == 'y':
    dump(model, 'ordinal_combined_models/logistic_combined_.joblib')
    print('Model save')
