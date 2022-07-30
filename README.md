# DATA310-Project-3
This project was compleated as a part of DATA 310 at William & Mary. The project consisted of one programming assignments. In addition to the README file, this repository contains a folder which has the .ipynb files for all questions.

## Question One

Question One asks the following: 

"For this project, you will apply the three grid search algorithms we studied, such as the Genetic Algorithm, Particle Swarm Optimization, and Simulated Annealing, to determine the best choice of hyperparameters for the Elastic Net and Support Vector Regressor method. Specifically, in the case of the Elastic Net, we look for the best alpha and l1_ratio, and in the case of SVR, we search for the best combination of epsilon and C values. The metric for decision is a 10-fold cross-validated MSE (external validation)."

For this project, I imported a given csv file using the following code.
```Python
df = pd.read_csv('drive/MyDrive/DATA 310/Datasets/AirI.csv',header=None)
Xf = df.values
X = Xf[:,1:]
y = Xf[:,0]
```

## Genetic Algorithm and Simulated Annealing
### Functions
From there I wrote two objective functions. They looked similar to each other, with the only difference being in the model line. 

This block is the objective function for the Elastic Net method:
```Python
def objective_ENet(h): # h is a two column matrix
  kf = KFold(n_splits=10,shuffle=True,random_state=123)
  scale = StandardScaler()
  output = [] 
  a = h[0]
  l = h[1]
  model = ElasticNet(alpha=a,l1_ratio=l,max_iter=5000)
  PE = []
  for idxtrain, idxtest in kf.split(X):
    Xtrain = X[idxtrain]
    ytrain = y[idxtrain]
    Xtrain_s = scale.fit_transform(Xtrain)
    Xtest = X[idxtest]
    ytest = y[idxtest]
    Xtest_s = scale.transform(Xtest)
    model.fit(Xtrain_s,ytrain)
    PE.append(MSE(ytest,model.predict(Xtest_s)))
  return np.mean(PE)
```

This block is the objective function for the Support Vector Regressor Method:
```Python
def objective_SVR(h): # h is a two column matrix
  kf = KFold(n_splits=10,shuffle=True,random_state=123)
  scale = StandardScaler()
  output = [] 
  Reg_Param = h[0] # column matrix 
  e = h[1]
  model = SVR(C=Reg_Param,epsilon=e,max_iter=5000)
  PE = []
  for idxtrain, idxtest in kf.split(X):
    Xtrain = X[idxtrain]
    ytrain = y[idxtrain]
    Xtrain_s = scale.fit_transform(Xtrain)
    Xtest = X[idxtest]
    ytest = y[idxtest]
    Xtest_s = scale.transform(Xtest)
    model.fit(Xtrain_s,ytrain)
    PE.append(MSE(ytest,model.predict(Xtest_s)))
  return np.mean(PE)
```

### Genetic Algorithm Testing and Results
After validating that thes objective functions were able to execute without errors, I turned my attention to executing the model. I used the following block of code. The array of h values was based on the work on ElasticNet and Support Vector Regression I did for project #2.
```Python
varbound=np.array([[0,3],[0,1]])
model=ga(function=objective_ENet,dimension=2,variable_type='real',variable_boundaries=varbound)
model.run()
```
This resulted in the optimal hyperparameters of alpha = 1.0017 and l1_ratio = 0.0048. At this point, the Mean Squared Error = 0.2578.

Similar to the objective functions, the Support Vector Regressor method was tested on the Genetic Algorithm in a very similar way.
```Python
varbound=np.array([[0,3],[0,10]])
model=ga(function=objective_SVR,dimension=2,variable_type='real',variable_boundaries=varbound)
model.run()
```
This resulted in the optimal hyperparameters of alpha = 2.4401 and l1_ratio = 0.0019. At this point, the Mean Squared Error = 0.3799.

### 

