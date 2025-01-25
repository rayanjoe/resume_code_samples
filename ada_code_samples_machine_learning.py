Table = [] 
Table_All = [] 
Table_All_c = []
Model = []
Score = []
best_model_score = 0 
Accuracy = []
score_g = make_scorer(f1_score, average='weighted')
def update_best_model(current_model, current_score, best_model, best_score):
    if current_score > best_score:
        best_model = current_model
        best_score = current_score
    return best_model, best_score 
def grid(mdl, grid_search_param, X_train, y_train, X_val, y_val, score):
    warnings.filterwarnings("ignore") 
    grid_search = GridSearchCV(mdl, grid_search_param, cv=5)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_ 
    best_score = grid_search.best_score_
    best_param = grid_search.best_params_
    if score == True:
        y_pred_valt = grid_search.predict(X_val)
        score_val = f1_score(y_val, y_pred_valt, average='weighted')
    else:
        score_val = 0 
        y_pred_valt = 0 
    return best_model, best_score, best_param, y_pred_valt, score_val, grid_search 
def specificity_(cf):
    tn = cf[0][0]
    fn = cf[1][0]
    specificity = tn / (tn+fn)
    return specificity
def print_(data_type, cmatrix, train_accuracy, test_accuracy, recall, specificity, precision, false_positive_rate, f1, score):
    print(data_type, "Confusion Marix:\n", cmatrix)
    print(data_type, "Training accuracy:", train_accuracy)
    print(data_type, "Testing accuracy:", test_accuracy)
    print(data_type, "Recall (Sensitivity):", recall)
    print(data_type, "Specificity:", specificity) 
    print(data_type, "Precision:", precision)
    print(data_type, "False Positive Rate:", false_positive_rate)
    print(data_type, "F1 Score:", f1)
    print(data_type, "Score:", score)
    print("#########################################################") 
    return None
def test(y_pred, model, X_train, X_test, y_train, y_test, val):
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, y_pred)
    cmatrix = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred, pos_label=1)
    specificity = specificity_(cmatrix)
    precision = precision_score(y_test, y_pred)
    false_positive_rate = 1 - specificity
    f1 = f1_score(y_test, y_pred)
    score = test_accuracy
    if val is False:
        print_("Test Data Set", cmatrix, train_accuracy, test_accuracy, recall, specificity, precision, false_positive_rate, f1, score)
    else:
        print_("Validation Data Set", cmatrix, train_accuracy, test_accuracy, recall, specificity, precision, false_positive_rate, f1, score)
    return cmatrix, train_accuracy, test_accuracy, recall, specificity, precision, false_positive_rate, f1, score

def print__(data_type, r2_train, rmse_train, r2_test, rmse_test):
    print(data_type, "R^2", r2_train)
    print(data_type, "R^2 on test", r2_test)
    print(data_type, "RMSE", rmse_train)
    print(data_type, "RMSE o n test", rmse_test)
    print("#########################################################") 
    return None

def test_(y_pred, model, X_train, X_test, y_train, y_test, val):
    #y_pred_train = model.predict(X_train)
    r2_train = r2_score(y_train, y_pred) 
    r2_test = r2_score(y_test, y_pred)
    mse_test = mean_squared_error(y_test, y_pred) 
    mse_train = mean_squared_error(y_train, y_pred)
    rmse_train = np.sqrt(mse_train) 
    rmse_test = np.sqrt(mse_test)  
    if val is False:  
        print__("Test Data Set", r2_train, rmse_train, r2_test, rmse_test) 
    else: 
        print__("Validation Data Set", r2_train, rmse_train, r2_test, rmse_test) 
    return r2_train, rmse_train 

#r2_train, rmse_train = model_L(X_train, X_test, y_train, y_test, model,  valid = None, X_val = X_val, y_val = y_val, nam = "LinearRegression")
def model_L(X_train, X_test, y_train, y_test, model, valid, X_val, y_val, nam):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2_train, rmse_train = test_(y_pred, model, X_train, X_test, y_train, y_test, val = False)
    #Table.append([nam+'_Test_Data_Set', r2_train, rmse_train])
    Table_All.append([nam+'_Test_Data_Set', r2_train, rmse_train, model])
    y_val_pred = model.predict(X_val) 
    r2_train, rmse_train = test_(y_val_pred, model, X_train, X_val, y_train = X_val, y_test = y_val, val = True)
    #Table.append([nam+'_VALIDATION_Data_Set', r2_train, rmse_train])
    Table_All.append([nam+'VALIDATION_Data_Set', r2_train, rmse_train, model])
    Model.append(model)
    #Score.append(score)
    if valid == 'CV':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    else: 
        cv_scores = 0 
    return  r2_train, rmse_train
 
def model_(X_train, X_test, y_train, y_test, model, valid, X_val, y_val, nam):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cmatrix, train_accuracy, test_accuracy, recall, specificity, precision, false_positive_rate, f1, score = test(y_pred, model, X_train, X_test, y_train, y_test, val = False)
    Table.append([nam+'_Test_Data_Set',model, score, recall, precision, f1 ])
    y_val_pred = model.predict(X_val) 
    cmatrix, train_accuracy, test_accuracy, recall, specificity, precision, false_positive_rate, f1, score = test(y_val_pred, model, X_train, X_val, y_train, y_val, val = True)
    Table.append([nam+'_VALIDATION_Data_Set',model, score, recall, precision, f1 ])
    Model.append(model)
    Score.append(score)
    if valid == 'CV':    
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    else:
        cv_scores = 0 
    return  cv_scores, score
