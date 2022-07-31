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
This resulted in the optimal hyperparameters of alpha ≈ 0.98704 and l1_ratio ≈ 0.00508. At this point, the Mean Squared Error ≈ 0.25785.

Similar to the objective functions, the Support Vector Regressor method was tested on the Genetic Algorithm in a very similar way.
```Python
varbound=np.array([[0,3],[0,10]])
model=ga(function=objective_SVR,dimension=2,variable_type='real',variable_boundaries=varbound)
model.run()
```
This resulted in the optimal hyperparameters of C ≈ 1.80854 and epsilon ≈ 0.13022. At this point, the Mean Squared Error ≈ 0.0.268588.

### Simulated Annealing Testing and Results

Simulated Annealing works with the same objective defined above. For ElasticNet, Simulated Annealing was tasked with identifying optimal alpha parameter and l1_ratio. For Support Vector Regression, the goal was to find the optimal Regularization Paramater (C) and epsilon value.
The code for ElasticNet is below:
```Python
lw = [0,0]
up = [3,1]
ret_ENet = dual_annealing(objective_ENet, bounds=list(zip(lw, up)),maxiter=10000,maxfun=10000)

ret_ENet
```
This resulted in the optimal hyperparameters of alpha ≈ 1.0001 and l1_ratio ≈ 0.0049. At this point, the Mean Squared Error ≈ 0.25782.

The Support Vector Algorithm was tested in a very similar way.
```Python
lw = [0.000001,0]
up = [10,10]
ret_SVR = dual_annealing(objective_SVR, bounds=list(zip(lw, up)),maxiter=10000,maxfun=10000)
```
This resulted in the optimal hyperparameters of C ≈ 3.14810 and epsilon ≈ 0.13079. At this point, the Mean Squared Error ≈ 0.268591.

## Partical Swarm
Partical Swarm Optimization relies on a different form of optimization function. It can be seen below:

```Python
def objective_pso_ENet(h):
  kf = KFold(n_splits=10,shuffle=True,random_state=123)
  scale = StandardScaler()
  output = [] 
  for i in range(h.shape[0]):
    a = h[i,0] # column matrix 
    l = h[i,1]
    model = ElasticNet(alpha=a,l1_ratio=l,max_iter=5000)
    PE = []
    for idxtrain, idxtest in kf.split(X):
      Xtrain = X[idxtrain]
      ytrain = y[idxtrain]
      Xtest = X[idxtest]
      ytest = y[idxtest]

      Xtrain_s = scale.fit_transform(Xtrain)
      Xtest_s = scale.transform(Xtest)
      model.fit(Xtrain_s,ytrain)
      PE.append(MSE(ytest,model.predict(Xtest_s)))
    output.append(np.mean(PE))
  return output
  ```
  Once again, the only difference between the above objective function and the one written for Support Vector Regression relates to the model line:
```Python  
def objective_pso_SVR(h):
  kf = KFold(n_splits=10,shuffle=True,random_state=123)
  scale = StandardScaler()
  output = [] 
  for i in range(h.shape[0]):
    Reg_Param = h[i,0] # column matrix 
    e = h[i,1]
    model = SVR(C=Reg_Param,epsilon=e,max_iter=5000)
    PE = []
    for idxtrain, idxtest in kf.split(X):
      Xtrain = X[idxtrain]
      ytrain = y[idxtrain]
      Xtest = X[idxtest]
      ytest = y[idxtest]

      Xtrain_s = scale.fit_transform(Xtrain)
      Xtest_s = scale.transform(Xtest)
      model.fit(Xtrain_s,ytrain)
      PE.append(MSE(ytest,model.predict(Xtest_s)))
    output.append(np.mean(PE))
  return output
  ```
### Partical Swarm Testing and Results

For ElasticNet, Partical Swarm was tasked with identifying optimal alpha parameter and l1_ratio. For Support Vector Regression, the goal was to find the optimal Regularization Paramater (C) and epsilon value.

The code for ElasticNet is:
```Python
max = np.array([3,1])
min = np.array([0,0])
bounds = (min, max)
options = {'c1': 0.25, 'c2': 0.25, 'w': 0.5}
optimizer = ps.single.GlobalBestPSO(n_particles=25, dimensions=2, options=options, bounds=bounds)
cost, pos = optimizer.optimize(objective_pso_ENet, iters=1000)
```
This resulted in the optimal hyperparameters of alpha ≈ 0.4008 and l1_ratio ≈ 0.1431. At this point, the Mean Squared Error ≈ 0.25978.

The code for Support Vector Regression is:
```Python
max = np.array([3,10])
min = np.array([0,0])
bounds = (min, max)
options = {'c1': 0.25, 'c2': 0.25, 'w': 0.5}
optimizer = ps.single.GlobalBestPSO(n_particles=25, dimensions=2, options=options, bounds=bounds)
cost, pos = optimizer.optimize(objective_pso_SVR, iters=1000)
```
This resulted in the optimal hyperparameters of C ≈ 0.70352 and epsilon ≈ 0.12682. At this point, the Mean Squared Error ≈ 0.268590.

## Identifying the best model
### ElasticNet
The results from the three tests of ElasticNet are summarized on the following datatable:
| Model  | Mean Squared Error |
| ------------- | ------------- |
| Genetic Algorithm  | 0.25785  |
| Simulated Annealing  | 0.25782 |
| Partical Swarm Optimization | 0.25978 |

Thefore, based on the external validation of the ElasticNet method, the best hyperparameters, as determined by the Simulated Annealing method are alpha ≈ 0.98704 and l1_ratio ≈ 0.00508.

### Support Vector Regression
| Model  | Mean Squared Error |
| ------------- | ------------- |
| Genetic Algorithm  |  0.268588 |
| Simulated Annealing  | 0.268591  |
| Partical Swarm Optimization | 0.268590 |

Thefore, based on the external validation of the Suppor Vector Regression method, the best hyperparameters, as determined by the Genetic Algorithm method are C ≈ 1.80854 and epsilon ≈ 0.13022.

## The Ridge and Lasso Extension
I chose to extend this project by using the above grid-search methods to attempt to replicate the results I got from project 2. This means identifying the optimal alpha hyperparameter for the Lasso and Ridge regressions methods. This process began by retrofitting my objective funtions such that they work find only one hyperparameter. They can be seen below:

## Objective Functions: Ridge and Lasso
### Ridge Regression
Ridge Objective Function:
```Python
def objective_Rid(h):
  kf = KFold(n_splits=10,shuffle=True,random_state=123)
  scale = StandardScaler()
  output = [] 
  a = h # column matrix 
  model = Ridge(alpha= a, fit_intercept = False, max_iter=5000)
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
Ridge Objective Function (Partical Swarm Optimization Version):

```Python
def objective_pso_Rid(h):
  kf = KFold(n_splits=10,shuffle=True,random_state=123)
  scale = StandardScaler()
  output = [] 
  for i in range(h.shape[0]):
    a = h[i,0]
    model = Ridge(alpha= a, fit_intercept = False,max_iter=5000)
    PE = []
    for idxtrain, idxtest in kf.split(X):
      Xtrain = X[idxtrain]
      ytrain = y[idxtrain]
      Xtest = X[idxtest]
      ytest = y[idxtest]

      Xtrain_s = scale.fit_transform(Xtrain)
      Xtest_s = scale.transform(Xtest)
      
      model.fit(Xtrain_s,ytrain)
      PE.append(MSE(ytest,model.predict(Xtest_s)))
    output.append(np.mean(PE))
  return output
```
### Lasso Regression
Ridge Objective Function:
```Python
def objective_Las(h):
  kf = KFold(n_splits=10,shuffle=True,random_state=123)
  scale = StandardScaler()
  output = [] 
  a = h
  model = Las(alpha= a, fit_intercept = False, max_iter=5000)
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
Ridge Objective Function (Partical Swarm Optimization Version):

```Python
def objective_pso_Las(h): # h is a two column matrix
  kf = KFold(n_splits=10,shuffle=True,random_state=123)
  scale = StandardScaler()
  output = [] 
  for i in range(h.shape[0]):
    a = h[i,0] # column matrix 
    model = Lasso(alpha= a, fit_intercept = False,max_iter=5000)
    PE = []
    for idxtrain, idxtest in kf.split(X):
      Xtrain = X[idxtrain]
      ytrain = y[idxtrain]
      Xtest = X[idxtest]
      ytest = y[idxtest]

      Xtrain_s = scale.fit_transform(Xtrain)
      Xtest_s = scale.transform(Xtest)
      
      model.fit(Xtrain_s,ytrain)
      PE.append(MSE(ytest,model.predict(Xtest_s)))
    output.append(np.mean(PE))
  return output
```

## Lasso Testing and Results

## Ridge Testing and Results
