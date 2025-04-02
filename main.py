import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import shapiro
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

#cd ER_ULAS /src/ 
#.\venv\Scripts\activate scrivere in terminal per attivare ambiente virtuale


df = pd.read_csv("Customer-Churn-Records.csv")

df.head()

df.info()

# 1.PREPROCESSING
df = df.dropna()

print(df.describe())

# rimuovo outliers
df = df[df['CreditScore'] < df['CreditScore'].quantile(0.95)]

# controllo
df.isnull().sum()

# non servono
df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# sto guardando tipi di geography nel dataset
Countries = df['Geography'].unique() 
print(Countries)

#sto guardando tipi di card type nel dataset
Types = df['Card Type'].unique()
print(Types)

#IMPORTANTE -> sto creando una pipeline per la standardizzazione delle variabili numeriche e one-hot encoding per quelle categoriali
numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'Point Earned', 'Satisfaction Score']
categorical_features = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember', 'Complain', 'Card Type']

numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[('num', numerical_transformer, numerical_features), ('cat', categorical_transformer, categorical_features)])

# cominciamo l'eda
#quanti customer sono usciti vs non usciti,conta 0 e 1
churn_counts = df['Exited'].value_counts()
colors = ['#7B68EE', '#483D8B']

#faccio un barchar
plt.figure(figsize=(8, 6))
plt.bar(churn_counts.index, churn_counts.values, color=colors)
plt.xlabel('Churn (Exited)')
plt.ylabel('Count')
plt.xticks(churn_counts.index, labels=['Not Churned', 'Churned'])
plt.title('Count of Customers Churned vs. Not Churned')
plt.show()

# faccio la matrice di correlazione
correlation_matrix = df[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# faccio un pairplot per osservare
sns.pairplot(df[['CreditScore', 'Age', 'Balance', 'Exited']], hue='Exited', palette='husl')
plt.show()

# faccio un boxplot per combinare churn e credit score
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Exited', y='CreditScore', palette='Set1')
plt.xlabel('Churn (Exited)')
plt.ylabel('CreditScore')
plt.title('CreditScore Distribution by Churn')
plt.xticks([0, 1], ['Not Churned', 'Churned'])
plt.show()


#SPLITTING
X = df.drop('Exited', axis=1)
y = df['Exited']

# 60% training, 20% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# creo il modello della regressione lineare(sapendo che il dataset non e' ottimale per una regressione lineare)
X_reg = df[['Age']]
y_reg = df['CreditScore']

reg_model = LinearRegression()
reg_model.fit(X_reg, y_reg)

# faccio la stima dei coefficienti
coef = reg_model.coef_[0]
intercept = reg_model.intercept_
print(f"Coefficiente: {coef:.4f}, Intercetta: {intercept:.4f}")

# predizioni per il grafico
y_reg_pred = reg_model.predict(X_reg)

# calcolo r2 e ms2
mse_reg = mean_squared_error(y_reg, y_reg_pred)
r2_reg = r2_score(y_reg, y_reg_pred)
print(f"Regression MSE: {mse_reg:.2f}, R^2: {r2_reg:.2f}")

# analisi di normalità dei residui (shapiro wilk test)
residuals = y_reg - y_reg_pred
stat, p_value = shapiro(residuals)
print(f'Shapiro-Wilk Test statistic: {stat:.4f}, p-value: {p_value:.4f}')

# qqplot per la normalità dei residui
plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ-Plot dei Residui")
plt.xlabel("Quantili Teorici")
plt.ylabel("Quantili dei Residui")
plt.grid(True)
plt.show()

# grafica scatter e retta di regressione
plt.figure(figsize=(8, 6))
plt.scatter(X_reg, y_reg, alpha=0.5, label='Dati')
plt.plot(X_reg, y_reg_pred, color='red', label='Retta di Regressione')
plt.xlabel('Age')
plt.ylabel('CreditScore')
plt.title('Linear Regression: Age vs CreditScore')
plt.legend()
plt.show()


# creo i modelli di svm e rl come pipeline
logistic_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

svm_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(random_state=42))
])



# ADDESTRAMENTO DEI MODELLI
logistic_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# HYPERMETER TUNING (Logistic Regression)
log_reg_params = {'classifier__C': [0.01, 0.1, 1, 10, 100]}
grid_search_log_reg = GridSearchCV(logistic_model, log_reg_params, cv=5, n_jobs=-1)
grid_search_log_reg.fit(X_val, y_val)
print(f'Miglior parametro C per la regressione logistica: {grid_search_log_reg.best_params_}')

# HYPERMETER TUNING SVM
svm_params = {
    'classifier__C': [0.01, 0.1],
    'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'classifier__gamma': ['scale', 'auto'],
    'classifier__degree': [2, 3, 4]
}
grid_search_svm = GridSearchCV(svm_model, svm_params, cv=3, n_jobs=-1)
grid_search_svm.fit(X_val, y_val)
print(f'Migliori parametri per SVM: {grid_search_svm.best_params_}')

# prediciamo il test set
y_pred_log_reg = grid_search_log_reg.predict(X_test)
y_pred_svm = grid_search_svm.predict(X_test)

# VALUTAZIONE DELLA PERFORMANCE
print("Valutazione della Regressione Logistica:")
print(classification_report(y_test, y_pred_log_reg))
print(confusion_matrix(y_test, y_pred_log_reg))

print("\nValutazione dell'SVM:")
print(classification_report(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))

# STUDIO STATISTICO,cross validation(10)
cv_scores_log_reg = cross_val_score(grid_search_log_reg.best_estimator_, X, y, cv=10)
print(f"Media dei punteggi di cross-validation per la regressione logistica: {cv_scores_log_reg.mean():.4f}")
print(f"Deviazione standard dei punteggi di cross-validation per la regressione logistica: {cv_scores_log_reg.std():.4f}")

cv_scores_svm = cross_val_score(grid_search_svm.best_estimator_, X, y, cv=10)
print(f"Media dei punteggi di cross-validation per SVM: {cv_scores_svm.mean():.4f}")
print(f"Deviazione standard dei punteggi di cross-validation per SVM: {cv_scores_svm.std():.4f}")

#boxplot per cross validation
plt.figure(figsize=(12, 8))
sns.boxplot(data=[cv_scores_log_reg, cv_scores_svm], orient='h')
plt.yticks([0, 1], ['Logistica', 'SVM'])
plt.title('Distribuzione dei Punteggi di Cross-Validation')
plt.xlabel('Accuracy')
plt.xlim(0.98, 1.0)  
plt.show()

# calcolo dell'intervallo di confidenza per la regressione logistica
confidence_interval_log_reg = stats.t.interval(0.95, len(cv_scores_log_reg)-1, loc=cv_scores_log_reg.mean(), scale=stats.sem(cv_scores_log_reg))
print(f"Intervallo di confidenza per la regressione logistica: {confidence_interval_log_reg}")

# calcolo dell'intervallo di confidenza per svm
confidence_interval_svm = stats.t.interval(0.95, len(cv_scores_svm)-1, loc=cv_scores_svm.mean(), scale=stats.sem(cv_scores_svm))
print(f"Intervallo di confidenza per SVM: {confidence_interval_svm}")




