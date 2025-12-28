import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import os

# Displaying the icon image
st.image("mentalhealth.jpg", use_column_width=True)
st.header("Mental Health Estimation Model Using Several Machine Learning Algorithm")
st.write("The Machine Learning Algorithm used in this Mental Health Estimation Model are Support Vector Machine, K-Nearest Neighbors, Decision Tree")

selectDataset = st.sidebar.selectbox ("Select Dataset", options = ["Select Dataset" ,"Stress level","Depression severity", "Schizophrenia", "PostMortem", "Panic Disorder"])

if selectDataset=="Stress level":
    st.image("stress.jpg", use_column_width=True)
    st.header("Stress Level")
    st.subheader ("Training dataset")
    training_data = pd.read_csv('sleep_training.csv')  
    training_data

    st.subheader ("Training for input and target")
    st.write("Data Input Training")
    data_input_traning=training_data.drop(columns='stress_level')
    data_input_traning

    st.write("Data Target Training")
    data_target_training = training_data['stress_level']
    data_target_training


    st.subheader ("Testing dataset")
    testing_data = pd.read_csv('sleep_testing.csv')  
    testing_data

    st.subheader ("Testing for input and target")
    st.write("Data Input Testing")
    data_input_testing = testing_data.drop(columns='stress_level')
    data_input_testing

    st.write("Data Target Testing")
    data_target_testing = testing_data['stress_level']
    data_target_testing

    selectModel = st.sidebar.selectbox("Select Model", options=["Select Model", "Support Vector Machine", "Decision Tree", "K-Nearest Neighbors"])

    if selectModel == "Support Vector Machine":
        st.title("Support Vector Machine Stress Estimation Model")
        st.subheader("Kernel Linear")

        svmLinearKernel = SVC(kernel = 'linear')

        st.write("Training the model...")
        svmLinearKernel.fit(data_input_traning, data_target_training)

        st.write("Successfully Train the Model")

        predictionSVMLinear = svmLinearKernel.predict(data_input_testing)

        st.write("Predicted Results for Testing Dataset:")
        predictionSVMLinear

        accuracyScoreSVMLinear = accuracy_score(predictionSVMLinear,data_target_testing)
        st.write("The Accuracy Score(Linear):",accuracyScoreSVMLinear)

        st.write("Predicting the target...")
        target_pred = svmLinearKernel.predict(data_input_testing)
        st.write("Successfully predict the target!")

        st.write("Linear Matrix")
        st.write(confusion_matrix(data_target_testing,target_pred))

        st.write("Classification Report")
        st.write(classification_report(data_target_testing,target_pred))

        st.subheader("Kernel Poly")

        svmPolyKernel = SVC(kernel = 'poly')

        st.write("Training the model...")
        svmPolyKernel.fit(data_input_traning, data_target_training)

        st.write("Successfully Train the Model")

        predictionSVMPoly = svmPolyKernel.predict(data_input_testing)

        st.write("Predicted Results for Testing Dataset:")
        predictionSVMPoly

        accuracyScoreSVMPoly = accuracy_score(predictionSVMPoly,data_target_testing)
        st.write("The Accuracy Score(Poly):",accuracyScoreSVMPoly)

        st.write("Predicting the target...")
        target_pred = svmPolyKernel.predict(data_input_testing)
        st.write("Successfully predict the target!")

        st.write("Poly Matrix")
        st.write(confusion_matrix(data_target_testing,target_pred))

        st.write("Classification Report")
        st.write(classification_report(data_target_testing,target_pred))

        st.subheader("Kernel Sigmoid")

        svmSigmoidKernel = SVC(kernel = 'sigmoid')

        st.write("Training the model...")
        svmSigmoidKernel.fit(data_input_traning, data_target_training)

        st.write("Successfully Train the Model")

        predictionSVMSigmoid = svmSigmoidKernel.predict(data_input_testing)

        st.write("Predicted Results for Testing Dataset:")
        predictionSVMSigmoid

        accuracyScoreSVMSigmoid = accuracy_score(predictionSVMSigmoid,data_target_testing)
        st.write("The Accuracy Score(Sigmoid):",accuracyScoreSVMSigmoid)

        st.write("Predicting the target...")
        target_pred = svmSigmoidKernel.predict(data_input_testing)
        st.write("Successfully predict the target!")

        st.write("Sigmoid Matrix")
        st.write(confusion_matrix(data_target_testing,target_pred))

        st.write("Classification Report")
        st.write(classification_report(data_target_testing,target_pred))
        
        st.subheader("Kernel RBF")

        svmRBFKernel = SVC(kernel = 'rbf')

        st.write("Training the model...")
        svmRBFKernel.fit(data_input_traning, data_target_training)

        st.write("Successfully Train the Model")

        predictionSVMRBF = svmRBFKernel.predict(data_input_testing)

        st.write("Predicted Results for Testing Dataset:")
        predictionSVMRBF

        accuracyScoreSVMRBF = accuracy_score(predictionSVMRBF,data_target_testing)
        st.write("The Accuracy Score(RBF):",accuracyScoreSVMRBF)

        st.write("Predicting the target...")
        target_pred = svmRBFKernel.predict(data_input_testing)
        st.write("Successfully predict the target!")

        st.write("RBF Matrix")
        st.write(confusion_matrix(data_target_testing,target_pred))

        st.write("Classification Report")
        st.write(classification_report(data_target_testing,target_pred))


    elif selectModel == "Decision Tree":

        st.title("Decision Tree Stress Level Estimation Model")
        dtClassifier  = DecisionTreeClassifier()

        st.write("Training the model...")
        dtClassifier.fit(data_input_traning,data_target_training)
        st.write("Successfully Train the Model")

        dt_results = dtClassifier.predict (data_input_testing)

        st.write("Predicted Results for Testing Dataset:")
        dt_results

        accuracy_dt = accuracy_score(dt_results, data_target_testing)
        st.write("Accuracy Score(Decision Tree):",accuracy_dt)

    elif selectModel == "K-Nearest Neighbors":

        st.title("K-Nearest Neighbors Stress Level Estimation Model")
        st.subheader("Number of Nearest Neighbors = 5")

        knn = KNeighborsClassifier(n_neighbors=5)

        st.write("Training the model...")
        knn.fit(data_input_traning, data_target_training)
        st.write("Successfully Train the Model")

        predictions_knn5 = knn.predict(data_input_testing)

        st.write("Predicted Results for Testing Dataset:")
        predictions_knn5

        accuracyScoreknn5 = accuracy_score(predictions_knn5, data_target_testing)
        st.write("The Accuracy Score(KNN=5):", accuracyScoreknn5)

        st.subheader("Number of Nearest Neighbors = 10")

        knn10 = KNeighborsClassifier(n_neighbors=10)

        st.write("Training the model...")
        knn10.fit(data_input_traning, data_target_training)
        st.write("Successfully Train the Model")

        predictions_knn10 = knn10.predict(data_input_testing)

        st.write("Predicted Results for Testing Dataset:")
        predictions_knn10

        accuracyScoreknn10 = accuracy_score(predictions_knn10, data_target_testing)
        st.write("The Accuracy Score(KNN=10):", accuracyScoreknn10)

        st.subheader("Number of Nearest Neighbors = 15")

        knn15 = KNeighborsClassifier(n_neighbors=15)

        st.write("Training the model...")
        knn15.fit(data_input_traning, data_target_training)
        st.write("Successfully Train the Model")

        predictions_knn15 = knn.predict(data_input_testing)

        st.write("Predicted Results for Testing Dataset:")
        predictions_knn15

        accuracyScoreknn15 = accuracy_score(predictions_knn15, data_target_testing)
        st.write("The Accuracy Score(KNN=15):", accuracyScoreknn15)

        st.subheader("Number of Nearest Neighbors = 20")

        knn20 = KNeighborsClassifier(n_neighbors=20)

        st.write("Training the model...")
        knn20.fit(data_input_traning, data_target_training)
        st.write("Successfully Train the Model")

        predictions_knn20 = knn20.predict(data_input_testing)

        st.write("Predicted Results for Testing Dataset:")
        predictions_knn20

        accuracyScoreknn20 = accuracy_score(predictions_knn20, data_target_testing)
        st.write("The Accuracy Score(KNN=20):", accuracyScoreknn20)

    displayResults = st.sidebar.selectbox("Display the Results", options=["Select Results", "Support Vector Machine", "Decision Tree", "K-Nearest Neighbors"])

    if displayResults == "Support Vector Machine":
            
        as_SVM_Stress = {'Kernel' : ['Linear', 'Polynomial', 'Sigmoid', 'RBF'], 'Accuracy Score' : [accuracyScoreSVMLinear, accuracyScoreSVMPoly, accuracyScoreSVMSigmoid, accuracyScoreSVMRBF] }
        tab_SVM_Stress = pd.DataFrame(as_SVM_Stress)
            
        st.subheader('Table of Accuracy Score Results of SVM Models')
        st.write(tab_SVM_Stress)
        st.write('Based on the table, Linear along side with Polynomial and RBF produce the best results of accuracy score which is 1 (100%). Thus, Linear kernel is chosen to represent SVM model in comparison with other algorithm.')
        
    elif displayResults == "Decision Tree":

        as_DT_Stress = {'Model': ['Decision Tree'], 'Accuracy Score': [accuracy_dt]}
        tab_DT_Stress = pd.DataFrame(as_DT_Stress)

        st.subheader('Table of Accuracy Score of Decision Tree Model')
        st.write(tab_DT_Stress)

    elif displayResults == "K-Nearest Neighbors":

        as_KNN_Stress = {'Number of Nearest Neighbors' : ['5','10', '15', '20'],'Accuracy Score': [accuracyScoreknn5, accuracyScoreknn10, accuracyScoreknn15, accuracyScoreknn20]}
        tab_KNN_Stress = pd.DataFrame(as_KNN_Stress)

        st.subheader('Table of Accuracy Score Results of KNN Models')
        st.write(tab_KNN_Stress)
        st.write('Based on the table above, all the Number of Nearest Neighbors produce 1 (100%) acuracy score. Thus, the Number of Nearest Neighbors 5 is chosen to represent KNN model in comparison with other algorithm.')

    
    conclusion = st.sidebar.selectbox("Conclusion", options=["Select conclusion", "Comparison of Three Models"])

    if conclusion == "Comparison of Three Models":

        st.subheader('Comparison of Results for SVM, KNN, and Decision Tree Models')
        st.write('Comparison of SVM, Decision Tree, and KNN for the Best Model of Stress Level Dataset')

        results_models = {'Model': ['SVM (kernel linear)', 'KNN (number of nearest neighbor = 5)', 'Decision Tree'],'Accuracy Score': ['1', '1', '0.9762']}
        tab_result = pd.DataFrame(results_models)
            
        st.write('Table of Comparison')
        st.write(tab_result)
        st.write('Based on the Table above, SVM and KNN  is producing the highest accuracy score in estimating the stress level, while the Decision Tree is the worst model. Thus, SVM and KNN is selected as the Best Model in estimating the stress level in this project.')


elif selectDataset=="Depression severity":
    st.image("d.jpg", use_column_width=True)
    st.header("Depression severity")
    st.subheader ("Training dataset")
    training_data = pd.read_csv('depression_training.csv')  
    training_data

    st.subheader ("Training for input and target")
    st.write("Data Input Training")
    data_input_training=training_data.drop(columns=["id","school_year","age","gender","bmi","who_bmi","phq_score","depression_severity","DEPRESSION_SEVERITY","depressiveness","suicidal","depression_diagnosis","depression_treatment","anxiety_severity","anxiousness","anxiety_diagnosis","anxiety_treatment","sleepiness"])
    data_input_training

    st.write("Data Target Training")
    data_target_training = training_data['DEPRESSION_SEVERITY']
    data_target_training


    st.subheader ("Testing dataset")
    testing_data= pd.read_csv('depression_testing.csv')  
    testing_data

    st.subheader ("Testing for input and target")
    st.write("Data Input Testing")
    data_input_testing=testing_data.drop(columns=["id","school_year","age","gender","bmi","who_bmi","phq_score","depression_severity","DEPRESSION_SEVERITY","depressiveness","suicidal","depression_diagnosis","depression_treatment","anxiety_severity","anxiousness","anxiety_diagnosis","anxiety_treatment","sleepiness"])
    data_input_testing

    st.write("Data Target Testing")
    data_target_testing = testing_data['DEPRESSION_SEVERITY']
    data_target_testing

    selectModel = st.sidebar.selectbox("Select Model", options=["Select Model", "Support Vector Machine", "Decision Tree", "K-Nearest Neighbors"])

    if selectModel == "Support Vector Machine":
        st.title("Support Vector Machine Stress Estimation Model")
        st.subheader("Kernel Linear")

        svmLinearKernel = SVC(kernel = 'linear')

        st.write("Training the model...")
        svmLinearKernel.fit(data_input_training, data_target_training)

        st.write("Successfully Train the Model")

        predictionSVMLinear = svmLinearKernel.predict(data_input_testing)

        st.write("Predicted Results for Testing Dataset:")
        predictionSVMLinear

        RaccuracyScoreSVMLinear = accuracy_score(predictionSVMLinear,data_target_testing)
        st.write("The Accuracy Score(Linear):",RaccuracyScoreSVMLinear)

        st.write("Predicting the target...")
        target_pred = svmLinearKernel.predict(data_input_testing)
        st.write("Successfully predict the target!")

        st.subheader("Kernel Poly")

        svmPolyKernel = SVC(kernel = 'poly')

        st.write("Training the model...")
        svmPolyKernel.fit(data_input_training, data_target_training)

        st.write("Successfully Train the Model")

        predictionSVMPoly = svmPolyKernel.predict(data_input_testing)

        st.write("Predicted Results for Testing Dataset:")
        predictionSVMPoly

        RaccuracyScoreSVMPoly = accuracy_score(predictionSVMPoly,data_target_testing)
        st.write("The Accuracy Score(Poly):",RaccuracyScoreSVMPoly)

        st.write("Predicting the target...")
        target_pred = svmPolyKernel.predict(data_input_testing)
        st.write("Successfully predict the target!")

        st.subheader("Kernel Sigmoid")

        svmSigmoidKernel = SVC(kernel = 'sigmoid')

        st.write("Training the model...")
        svmSigmoidKernel.fit(data_input_training, data_target_training)

        st.write("Successfully Train the Model")

        predictionSVMSigmoid = svmSigmoidKernel.predict(data_input_testing)

        st.write("Predicted Results for Testing Dataset:")
        predictionSVMSigmoid

        RaccuracyScoreSVMSigmoid = accuracy_score(predictionSVMSigmoid,data_target_testing)
        st.write("The Accuracy Score(Sigmoid):",RaccuracyScoreSVMSigmoid)

        st.write("Predicting the target...")
        target_pred = svmSigmoidKernel.predict(data_input_testing)
        st.write("Successfully predict the target!")
        
        st.subheader("Kernel RBF")

        svmRBFKernel = SVC(kernel = 'rbf')

        st.write("Training the model...")
        svmRBFKernel.fit(data_input_training, data_target_training)

        st.write("Successfully Train the Model")

        predictionSVMRBF = svmRBFKernel.predict(data_input_testing)

        st.write("Predicted Results for Testing Dataset:")
        predictionSVMRBF

        RaccuracyScoreSVMRBF = accuracy_score(predictionSVMRBF,data_target_testing)
        st.write("The Accuracy Score(RBF):",RaccuracyScoreSVMRBF)

        st.write("Predicting the target...")
        target_pred = svmRBFKernel.predict(data_input_testing)
        st.write("Successfully predict the target!")


    elif selectModel == "Decision Tree":

        st.title("Decision Tree Stress Level Estimation Model")
        dtClassifier = DecisionTreeClassifier()

        st.write("Training the model...")
        dtClassifier.fit(data_input_training,data_target_training)
        st.write("Successfully Train the Model")

        dt_results = dtClassifier.predict (data_input_testing)

        st.write("Predicted Results for Testing Dataset:")
        dt_results

        Raccuracy_dt = accuracy_score(dt_results, data_target_testing)
        st.write("Accuracy Score(Decision Tree):",Raccuracy_dt)

    elif selectModel == "K-Nearest Neighbors":

        st.title("K-Nearest Neighbors Stress Level Estimation Model")
        st.subheader("Number of Nearest Neighbors = 5")

        knn = KNeighborsClassifier(n_neighbors=5)

        st.write("Training the model...")
        knn.fit(data_input_training, data_target_training)
        st.write("Successfully Train the Model")

        predictions_knn5 = knn.predict(data_input_testing)

        st.write("Predicted Results for Testing Dataset:")
        predictions_knn5

        RaccuracyScoreknn5 = accuracy_score(predictions_knn5, data_target_testing)
        st.write("The Accuracy Score(KNN=5):", RaccuracyScoreknn5)

        st.subheader("Number of Nearest Neighbors = 10")

        knn10 = KNeighborsClassifier(n_neighbors=10)

        st.write("Training the model...")
        knn10.fit(data_input_training, data_target_training)
        st.write("Successfully Train the Model")

        predictions_knn10 = knn10.predict(data_input_testing)

        st.write("Predicted Results for Testing Dataset:")
        predictions_knn10

        RaccuracyScoreknn10 = accuracy_score(predictions_knn10, data_target_testing)
        st.write("The Accuracy Score(KNN=10):", RaccuracyScoreknn10)

        st.subheader("Number of Nearest Neighbors = 15")

        knn15 = KNeighborsClassifier(n_neighbors=15)

        st.write("Training the model...")
        knn15.fit(data_input_training, data_target_training)
        st.write("Successfully Train the Model")

        predictions_knn15 = knn15.predict(data_input_testing)

        st.write("Predicted Results for Testing Dataset:")
        predictions_knn15

        RaccuracyScoreknn15 = accuracy_score(predictions_knn15, data_target_testing)
        st.write("The Accuracy Score(KNN=15):", RaccuracyScoreknn15)

        st.subheader("Number of Nearest Neighbors = 20")

        knn20 = KNeighborsClassifier(n_neighbors=20)

        st.write("Training the model...")
        knn20.fit(data_input_training, data_target_training)
        st.write("Successfully Train the Model")

        predictions_knn20 = knn20.predict(data_input_testing)

        st.write("Predicted Results for Testing Dataset:")
        predictions_knn20

        RaccuracyScoreknn20 = accuracy_score(predictions_knn20, data_target_testing)
        st.write("The Accuracy Score(KNN=20):", RaccuracyScoreknn20)

    displayResults = st.sidebar.selectbox("Display the Results", options=["Select Results", "Support Vector Machine", "Decision Tree", "K-Nearest Neighbors"])

    if displayResults == "Support Vector Machine":
            
        as_SVM_Depression = {'Kernel' : ['Linear', 'Polynomial', 'Sigmoid', 'RBF'], 'Accuracy Score' : [RaccuracyScoreSVMLinear, RaccuracyScoreSVMPoly, RaccuracyScoreSVMSigmoid, RaccuracyScoreSVMRBF] }
        tab_SVM_Depression = pd.DataFrame(as_SVM_Depression)
            
        st.subheader('Table of Accuracy Score Results of SVM Models')
        st.write(tab_SVM_Depression)
        st.write('Based on the table above, Linear kernel produces the highest accuracy score which is 0.7452 (74.52%) compare to other kernel. Thus, Linear kernel is chosen to represent SVM model in comparison with other algorithm.')
        
    elif displayResults == "Decision Tree":

        as_DT_Depression = {'Model': ['Decision Tree'], 'Accuracy Score': [Raccuracy_dt]}
        tab_DT_Depression = pd.DataFrame(as_DT_Depression)

        st.subheader('Table of Accuracy Score of Decision Tree Model')
        st.write(tab_DT_Depression)

    elif displayResults == "K-Nearest Neighbors":

        as_KNN_Depression = {'Number of Nearest Neighbors' : ['5','10', '15', '20'],'Accuracy Score': [RaccuracyScoreknn5, RaccuracyScoreknn10, RaccuracyScoreknn15, RaccuracyScoreknn20]}
        tab_KNN_Depression = pd.DataFrame(as_KNN_Depression)

        st.subheader('Table of Accuracy Score Results of KNN Models')
        st.write(tab_KNN_Depression)
        st.write('Based on the table above, the Number of Nearest Neighbors 15 produces the highest acuracy score which is 0.5732 (57.32%). Thus, the Number of Nearest Neighbors 15 is chosen to represent KNN model in comparison with other algorithm.')


    conclusion = st.sidebar.selectbox("Conclusion", options=["Select conclusion", "Comparison of Three Models"])

    if conclusion == "Comparison of Three Models":

        st.subheader('Comparison of Results for SVM, KNN, and Decision Tree Models')
        st.write('Comparison of SVM, Decision Tree, and KNN for the Best Model of Depression Severity Dataset')

        results_models = {'Model': ['SVM (kernel linear)', 'KNN (number of nearest neighbor = 15)', 'Decision Tree'],'Accuracy Score': ['0.7452', '0.5732', '0.6369']}
        tab_result = pd.DataFrame(results_models)
            
        st.write('Table of Comparison')
        st.write(tab_result)
        st.write('Based on the Table above, SVM model produces the best result of highest accuracy score in estimating the depression severity, while the KNN is the worst model. Thus, SVM is selected as the Best Model in estimating the depression severity in this project.')

elif selectDataset == "Schizophrenia":
    st.image("s.jpg", use_column_width=True)
    st.header("Schizophrenia")
    st.subheader("Training dataset")
    train_data = pd.read_csv('SchizophreniaTrain.csv')
    train_data

    st.subheader("Training for input and target")
    st.write("Data Input Training")
    input_train_data = train_data.drop(columns=['Schizophrenia', 'Name', 'Gender', 'Marital_Status'])
    nan_indices = pd.isna(input_train_data).any(axis=1)
    input_train_data = input_train_data.loc[~nan_indices, :]
    target_train_data = train_data['Schizophrenia'].values.ravel()
    target_train_data = target_train_data[~nan_indices]
    input_train_data

    st.write("Data Target Training")
    target_train_data

    st.subheader("Testing dataset")
    test_data = pd.read_csv('SchizophreniaTest.csv')
    test_data

    st.subheader("Testing for input and target")
    st.write("Data Input Testing")
    input_test_data = test_data.drop(columns=['Schizophrenia', 'Name', 'Gender', 'Marital_Status'])
    nan_indices = pd.isna(input_test_data).any(axis=1)
    input_test_data = input_test_data.loc[~nan_indices, :]
    target_test_data = test_data['Schizophrenia'].values.ravel()
    target_test_data = target_test_data[~nan_indices]
    input_test_data

    st.write("Data Target Testing")
    target_test_data

    selectModel = st.sidebar.selectbox("Select Model", options=["Select Model", "Support Vector Machine", "Decision Tree", "K-Nearest Neighbors"])

    if selectModel == "Support Vector Machine":
        st.title("Support Vector Machine Stress Estimation Model")
        st.subheader("Kernel Linear")

        SVCmodel = SVC(kernel='linear')

        st.write("Training the model...")
        SVCmodel.fit(input_train_data, target_train_data)

        st.write("Successfully Train the Model")

        resultsLinear = SVCmodel.predict(input_test_data)

        st.write("Predicted Results for Testing Dataset:")
        resultsLinear

        SaccuracyScoreLinear = accuracy_score(resultsLinear, target_test_data)
        st.write("The Accuracy Score(Linear):", SaccuracyScoreLinear)

        st.subheader("Kernel Poly")

        SVCmodel = SVC(kernel='poly')

        st.write("Training the model...")
        SVCmodel.fit(input_train_data, target_train_data)

        st.write("Successfully Train the Model")

        resultsPoly = SVCmodel.predict(input_test_data)

        st.write("Predicted Results for Testing Dataset:")
        resultsPoly

        accuracyScorePoly = accuracy_score(resultsPoly, target_test_data)
        st.write("The Accuracy Score(Poly):", accuracyScorePoly)

        st.subheader("Kernel Sigmoid")

        SVCmodel = SVC(kernel='sigmoid')

        st.write("Training the model...")
        SVCmodel.fit(input_train_data, target_train_data)

        st.write("Successfully Train the Model")

        resultsSigmoid = SVCmodel.predict(input_test_data)

        st.write("Predicted Results for Testing Dataset:")
        resultsSigmoid

        accuracyScoreSigmoid = accuracy_score(resultsSigmoid, target_test_data)
        st.write("The Accuracy Score(Sigmoid):", accuracyScoreSigmoid)

        st.subheader("Kernel RBF")

        SVCmodel = SVC(kernel='rbf')

        st.write("Training the model...")
        SVCmodel.fit(input_train_data, target_train_data)

        st.write("Successfully Train the Model")

        resultsRbf = SVCmodel.predict(input_test_data)

        st.write("Predicted Results for Testing Dataset:")
        resultsRbf

        NaccuracyScoreRbf = accuracy_score(resultsRbf, target_test_data)
        st.write("The Accuracy Score(RBF):", NaccuracyScoreRbf)

    elif selectModel == "Decision Tree":

        st.title("Decision Tree Stress Level Estimation Model")
        dtClassifier = DecisionTreeClassifier()

        st.write("Training the model...")
        dtClassifier.fit(input_train_data, target_train_data)
        st.write("Successfully Train the Model")

        resultsDt = dtClassifier.predict(input_test_data)

        st.write("Predicted Results for Testing Dataset:")
        resultsDt

        accuracyScoreDt = accuracy_score(resultsDt, target_test_data)
        st.write("Accuracy Score(Decision Tree):", accuracyScoreDt)

    elif selectModel == "K-Nearest Neighbors":

        st.title("K-Nearest Neighbors Stress Level Estimation Model")
        st.subheader("Number of Nearest Neighbors = 5")

        knn5 = KNeighborsClassifier(n_neighbors=5)

        st.write("Training the model...")
        knn5.fit(input_train_data, target_train_data)
        st.write("Successfully Train the Model")

        prediction5 = knn5.predict(input_test_data.values)

        st.write("Predicted Results for Testing Dataset:")
        prediction5

        accuracy_knn5 = accuracy_score(prediction5, target_test_data)
        st.write("The Accuracy Score(KNN=5):", accuracy_knn5)

        st.write("Predicting the target...")
        target_pred5 = knn5.predict(input_test_data.values)
        st.write("Successfully predict the target!")

        st.write("KNN5 Matrix")
        st.write(confusion_matrix(target_test_data, target_pred5))

        st.write("Classification Report")
        st.write(classification_report(target_test_data, target_pred5))

        st.subheader("Number of Nearest Neighbors = 10")

        knn10 = KNeighborsClassifier(n_neighbors=10)

        st.write("Training the model...")
        knn10.fit(input_train_data, target_train_data)
        st.write("Successfully Train the Model")

        prediction10 = knn10.predict(input_test_data.values)

        st.write("Predicted Results for Testing Dataset:")
        prediction10

        accuracy_knn10 = accuracy_score(prediction10, target_test_data)
        st.write("The Accuracy Score(KNN=10):", accuracy_knn10)

        st.subheader("Number of Nearest Neighbors = 15")

        knn15 = KNeighborsClassifier(n_neighbors=15)

        st.write("Training the model...")
        knn15.fit(input_train_data, target_train_data)
        st.write("Successfully Train the Model")

        prediction15 = knn15.predict(input_test_data.values)

        st.write("Predicted Results for Testing Dataset:")
        prediction15

        accuracy_knn15 = accuracy_score(prediction15, target_test_data)
        st.write("The Accuracy Score(KNN=15):", accuracy_knn15)

        st.subheader("Number of Nearest Neighbors = 20")

        knn20 = KNeighborsClassifier(n_neighbors=20)

        st.write("Training the model...")
        knn20.fit(input_train_data, target_train_data)
        st.write("Successfully Train the Model")

        prediction20 = knn20.predict(input_test_data.values)

        st.write("Predicted Results for Testing Dataset:")
        prediction20

        accuracy_knn20 = accuracy_score(prediction20, target_test_data)
        st.write("The Accuracy Score(KNN=20):", accuracy_knn20)

    displayResults = st.sidebar.selectbox("Display the Results", options=["Select Results", "Support Vector Machine", "Decision Tree", "K-Nearest Neighbors"])

    if displayResults == "Support Vector Machine":
            
        as_SVM_Schizo = {'Kernel' : ['Linear', 'Polynomial', 'Sigmoid', 'RBF'], 'Accuracy Score' : [SaccuracyScoreLinear, accuracyScorePoly, accuracyScoreSigmoid, NaccuracyScoreRbf] }
        tab_SVM_Schizo = pd.DataFrame(as_SVM_Schizo)
            
        st.subheader('Table of Accuracy Score Results of SVM Models')
        st.write(tab_SVM_Schizo)
        st.write('Based on the table above, Linear kernel produces the highest accuracy score which is 0.9722 (97.22%) compare to other kernel. Thus, Linear kernel is chosen to represent SVM model in comparison with other algorithm.')
        
    elif displayResults == "Decision Tree":

        as_DT_Schizo = {'Model': ['Decision Tree'], 'Accuracy Score': [accuracyScoreDt]}
        tab_DT_Schizo = pd.DataFrame(as_DT_Schizo)

        st.subheader('Table of Accuracy Score of Decision Tree Model')
        st.write(tab_DT_Schizo)

    elif displayResults == "K-Nearest Neighbors":

        as_KNN_Schizo = {'Number of Nearest Neighbors' : ['5','10', '15', '20'],'Accuracy Score': [accuracy_knn5, accuracy_knn10, accuracy_knn15, accuracy_knn20]}
        tab_KNN_Schizo = pd.DataFrame(as_KNN_Schizo)

        st.subheader('Table of Accuracy Score Results of KNN Models')
        st.write(tab_KNN_Schizo)
        st.write('Based on the table above, the Number of Nearest Neighbors 5 produces the highest acuracy score which is 0.7717 (77.17%). Thus, the Number of Nearest Neighbors 5 is chosen to represent KNN model in comparison with other algorithm.')

    conclusion = st.sidebar.selectbox("Conclusion", options=["Select conclusion", "Comparison of Three Models"])

    if conclusion == "Comparison of Three Models":

        st.subheader('Comparison of Results for SVM, KNN, and Decision Tree Models')
        st.write('Comparison of SVM, Decision Tree, and KNN for the Best Model of Schizophrenia Dataset')

        results_models = {'Model': ['SVM (kernel poly)', 'KNN (number of nearest neighbor = 5)', 'Decision Tree'],'Accuracy Score': ['0.9722.', '0.7717', '0.7822']}
        tab_result = pd.DataFrame(results_models)
            
        st.write('Table of Comparison')
        st.write(tab_result)
        st.write('Based on the Table above, SVM model produces the best result of highest accuracy score in estimating schizophrenia symptoms, while the KNN is the worst model. Thus, SVM is selected as the Best Model in estimating schizophrenia symptoms in this project.')




elif selectDataset == "PostMortum":
    st.image("postmartum.jpg", use_column_width=True)
    st.header("PostMortum")
    st.subheader("Training dataset")
    training_data = pd.read_csv('training_2post.csv')
    training_data

    st.subheader("Training for input and target")
    st.write("Data Input Training")
    data_input_training = training_data.drop(columns=['Timestamp', 'Age'])
    data_input_training

    st.write("Data Target Training")
    data_target_training = training_data['Age']
    data_target_training

    st.subheader("Testing dataset")
    test_data = pd.read_csv('testing_2post.csv')
    test_data

    st.subheader("Testing for input and target")
    st.write("Data Input Testing")
    data_input_testing = test_data.drop(columns=['Timestamp', 'Age'])
    data_input_testing

    st.write("Data Target Testing")
    data_target_testing = test_data['Age']
    data_target_testing

    selectModel = st.sidebar.selectbox("Select Model", options=["Select Model", "Support Vector Machine", "Decision Tree", "K-Nearest Neighbors"])

    if selectModel == "Support Vector Machine":
        st.title("Support Vector Machine Stress Estimation Model")
        st.subheader("Kernel Linear")

        svmLinearKernel = SVC(kernel='linear')

        st.write("Training the model...")
        svmLinearKernel.fit(data_input_training, data_target_training)

        st.write("Successfully Train the Model")

        predictionSVMLinear = svmLinearKernel.predict(data_input_testing)

        st.write("Predicted Results for Testing Dataset:")
        predictionSVMLinear

        accuracyScoreSVMLinear = accuracy_score(predictionSVMLinear, data_target_testing)
        st.write("The Accuracy Score(Linear):", accuracyScoreSVMLinear)

        st.subheader("Kernel Poly")

        svmPolyKernel = SVC(kernel='poly')

        st.write("Training the model...")
        svmPolyKernel.fit(data_input_training, data_target_training)

        st.write("Successfully Train the Model")

        predictionSVMPoly = svmPolyKernel.predict(data_input_testing)

        st.write("Predicted Results for Testing Dataset:")
        predictionSVMPoly

        accuracyScoreSVMPoly = accuracy_score(predictionSVMPoly, data_target_testing)
        st.write("The Accuracy Score(Poly):", accuracyScoreSVMPoly)

        st.subheader("Kernel Sigmoid")

        svmSigmoidKernel = SVC(kernel='sigmoid')

        st.write("Training the model...")
        svmSigmoidKernel.fit(data_input_training, data_target_training)

        st.write("Successfully Train the Model")

        predictionSVMSigmoid = svmSigmoidKernel.predict(data_input_testing)

        st.write("Predicted Results for Testing Dataset:")
        predictionSVMSigmoid

        accuracyScoreSVMSigmoid = accuracy_score(predictionSVMSigmoid, data_target_testing)
        st.write("The Accuracy Score(Sigmoid):", accuracyScoreSVMSigmoid)

        st.subheader("Kernel RBF")

        svmRbfKernel = SVC(kernel='rbf')

        st.write("Training the model...")
        svmRbfKernel.fit(data_input_training, data_target_training)

        st.write("Successfully Train the Model")

        predictionSVMRbf = svmRbfKernel.predict(data_input_testing)

        st.write("Predicted Results for Testing Dataset:")
        predictionSVMRbf

        accuracyScoreSVMRbf = accuracy_score(predictionSVMRbf, data_target_testing)
        st.write("The Accuracy Score(RBF):", accuracyScoreSVMRbf)

    elif selectModel == "Decision Tree":

        st.title("Decision Tree Stress Level Estimation Model")
        dt = DecisionTreeClassifier()

        st.write("Training the model...")
        dt.fit(data_input_training, data_target_training)
        st.write("Successfully Train the Model")

        dt_results = dt.predict(data_input_testing)

        st.write("Predicted Results for Testing Dataset:")
        dt_results

        accuracyScoreDt = accuracy_score(dt_results, data_target_testing)
        st.write("Accuracy Score(Decision Tree):", accuracyScoreDt)

    elif selectModel == "K-Nearest Neighbors":

        st.title("K-Nearest Neighbors Stress Level Estimation Model")
        st.subheader("Number of Nearest Neighbors = 5")

        knn5 = KNeighborsClassifier(n_neighbors=5)

        st.write("Training the model...")
        knn5.fit(data_input_training, data_target_training)
        st.write("Successfully Train the Model")

        predictions_knn5 = knn5.predict(data_input_testing)

        st.write("Predicted Results for Testing Dataset:")
        predictions_knn5

        accuracyScoreknn5 = accuracy_score(predictions_knn5, data_target_testing)
        st.write("The Accuracy Score(KNN=5):", accuracyScoreknn5)

        st.subheader("Number of Nearest Neighbors = 10")

        knn10 = KNeighborsClassifier(n_neighbors=10)

        st.write("Training the model...")
        knn10.fit(data_input_training, data_target_training)
        st.write("Successfully Train the Model")

        predictions_knn10 = knn10.predict(data_input_testing)

        st.write("Predicted Results for Testing Dataset:")
        predictions_knn10

        accuracyScoreknn10 = accuracy_score(predictions_knn10, data_target_testing)
        st.write("The Accuracy Score(KNN=10):", accuracyScoreknn10)

        st.subheader("Number of Nearest Neighbors = 15")

        knn15 = KNeighborsClassifier(n_neighbors=15)

        st.write("Training the model...")
        knn15.fit(data_input_training, data_target_training)
        st.write("Successfully Train the Model")

        predictions_knn15 = knn15.predict(data_input_testing)

        st.write("Predicted Results for Testing Dataset:")
        predictions_knn15

        accuracyScoreknn15 = accuracy_score(predictions_knn15, data_target_testing)
        st.write("The Accuracy Score(KNN=15):", accuracyScoreknn15)

        st.subheader("Number of Nearest Neighbors = 20")

        knn20 = KNeighborsClassifier(n_neighbors=20)

        st.write("Training the model...")
        knn20.fit(data_input_training, data_target_training)
        st.write("Successfully Train the Model")

        predictions_knn20 = knn20.predict(data_input_testing)
        predictions_knn20

        st.write("Predicted Results for Testing Dataset:")
        predictions_knn20

        accuracyScoreknn20 = accuracy_score(predictions_knn20, data_target_testing)
        st.write("The Accuracy Score(KNN=20):", accuracyScoreknn20)

    displayResults = st.sidebar.selectbox("Display the Results", options=["Select Results", "Support Vector Machine", "Decision Tree", "K-Nearest Neighbors"])

    if displayResults == "Support Vector Machine":
            
        as_SVM_Mortem = {'Kernel' : ['Linear', 'Polynomial', 'Sigmoid', 'RBF'], 'Accuracy Score' : [accuracyScoreSVMLinear, accuracyScoreSVMPoly, accuracyScoreSVMSigmoid, accuracyScoreSVMRbf] }
        tab_SVM_Mortem = pd.DataFrame(as_SVM_Mortem)
            
        st.subheader('Table of Accuracy Score Results of SVM Models')
        st.write(tab_SVM_Mortem)
        st.write('Based on the table above, Polynomial kernel produces the highest accuracy score which is 0.1063 (10.63%) compare to other kernel. Thus, Polynomial kernel is chosen to represent SVM model in comparison with other algorithm.')
        
    elif displayResults == "Decision Tree":

        as_DT_Mortem = {'Model': ['Decision Tree'], 'Accuracy Score': [accuracyScoreDt]}
        tab_DT_Mortem = pd.DataFrame(as_DT_Mortem)

        st.subheader('Table of Accuracy Score of Decision Tree Model')
        st.write(tab_DT_Mortem)

    elif displayResults == "K-Nearest Neighbors":

        as_KNN_Mortem = {'Number of Nearest Neighbors' : ['5','10', '15', '20'],'Accuracy Score': [accuracyScoreknn5, accuracyScoreknn10, accuracyScoreknn15, accuracyScoreknn20]}
        tab_KNN_Mortem = pd.DataFrame(as_KNN_Mortem)

        st.subheader('Table of Accuracy Score Results of KNN Models')
        st.write(tab_KNN_Mortem)
        st.write('Based on the table above, the Number of Nearest Neighbors 5 and 15 produce the same score of highest acuracy score which is 0.0631 (6.31%). Thus, the Number of Nearest Neighbors 5 is chosen to represent KNN model in comparison with other algorithm.')

    conclusion = st.sidebar.selectbox("Conclusion", options=["Select conclusion", "Comparison of Three Models"])

    if conclusion == "Comparison of Three Models":

        st.subheader('Comparison of Results for SVM, KNN, and Decision Tree Models')
        st.write('Comparison of SVM, Decision Tree, and KNN for the Best Model of PostMortem Dataset')

        results_models = {'Model': ['SVM (kernel poly)', 'KNN (number of nearest neighbor = 5)', 'Decision Tree'],'Accuracy Score': ['0.1063', '0.0631', '0.113']}
        tab_result = pd.DataFrame(results_models)
            
        st.write('Table of Comparison')
        st.write(tab_result)
        st.write('Based on the Table above, Decision Tree model produces the best result of highest accuracy score in estimating age of postmortem women based on the symptoms, while the KNN is the worst model. Thus, Decision Tree is selected as the Best Model in estimating age of postmortem women based on the symptoms in this project.')



elif selectDataset == "Panic Disorder":
    st.image("panic.jpg", use_column_width=True)
    st.header("Panic Disorder")
    st.subheader("Training dataset")
    training_data = pd.read_csv('panic_disorder_dataset_training.csv')
    st.write(training_data)

    st.subheader("Training for input and target")
    st.write("Data Input Training")
    data_input_training = training_data.drop(columns=['Participant ID', 'Panic Disorder Diagnosis'])
    st.write(data_input_training)

    st.write("Data Target Training")
    data_target_training = training_data['Panic Disorder Diagnosis']
    st.write(data_target_training)

    st.subheader("Testing dataset")
    test_data = pd.read_csv('panic_disorder_dataset_testing1.csv')
    st.write(test_data)

    st.subheader("Testing for input and target")
    st.write("Data Input Testing")
    data_input_testing = test_data.drop(columns=['Participant ID', 'Panic Disorder Diagnosis'])
    st.write(data_input_testing)

    st.write("Data Target Testing")
    data_target_testing = test_data['Panic Disorder Diagnosis']
    st.write(data_target_testing)

    data_input_training1 = pd.get_dummies(data_input_training, columns=['Gender', 'Current Stressors', 'Symptoms', 'Severity', 'Impact on Life', 'Demographics', 'Medical History', 'Psychiatric History', 'Substance Use', 'Coping Mechanisms', 'Social Support', 'Lifestyle Factors'])
    data_input_testing1 = pd.get_dummies(data_input_testing, columns=['Gender', 'Current Stressors', 'Symptoms', 'Severity', 'Impact on Life', 'Demographics', 'Medical History', 'Psychiatric History', 'Substance Use', 'Coping Mechanisms', 'Social Support', 'Lifestyle Factors'])

    st.write("Updated Data Input Testing")
    st.write(data_input_testing1)

    selectModel = st.sidebar.selectbox("Select Model", options=["Select Model", "Support Vector Machine", "Decision Tree", "K-Nearest Neighbors"])

    if selectModel == "Support Vector Machine":
        st.title("Support Vector Machine Stress Estimation Model")
        st.subheader("Kernel Linear")

        svcKernelLinear = SVC(kernel='linear')

        st.write("Training the model...")
        svcKernelLinear.fit(data_input_training1, data_target_training)

        st.write("Successfully Train the Model")

        resultsLinear = svcKernelLinear.predict(data_input_testing1)

        st.write("Predicted Results for Testing Dataset:")
        st.write(resultsLinear)

        PPaccuracyScoreLinear = accuracy_score(resultsLinear, data_target_testing)
        st.write("The Accuracy Score(Linear):", PPaccuracyScoreLinear)

        st.subheader("Kernel Poly")

        svcKernelPoly = SVC(kernel='poly')

        st.write("Training the model...")
        svcKernelPoly.fit(data_input_training1, data_target_training)

        st.write("Successfully Train the Model")

        resultsPoly = svcKernelPoly.predict(data_input_testing1)

        st.write("Predicted Results for Testing Dataset:")
        st.write(resultsPoly)

        PPaccuracyScorePoly = accuracy_score(resultsPoly, data_target_testing)
        st.write("The Accuracy Score(Poly):", PPaccuracyScorePoly)

        st.subheader("Kernel Sigmoid")

        svcKernelSigmoid = SVC(kernel='sigmoid')

        st.write("Training the model...")
        svcKernelSigmoid.fit(data_input_training1, data_target_training)

        st.write("Successfully Train the Model")

        resultsSigmoid = svcKernelSigmoid.predict(data_input_testing1)

        st.write("Predicted Results for Testing Dataset:")
        st.write(resultsSigmoid)

        PPaccuracyScoreSigmoid = accuracy_score(resultsSigmoid, data_target_testing)
        st.write("The Accuracy Score(Sigmoid):", PPaccuracyScoreSigmoid)

        st.subheader("Kernel RBF")

        svcKernelRBF = SVC(kernel='rbf')

        st.write("Training the model...")
        svcKernelRBF.fit(data_input_training1, data_target_training)

        st.write("Successfully Train the Model")

        resultsRBF = svcKernelRBF.predict(data_input_testing1)

        st.write("Predicted Results for Testing Dataset:")
        st.write(resultsRBF)

        PPaccuracyScoreRBF = accuracy_score(resultsRBF, data_target_testing)
        st.write("The Accuracy Score(RBF):", PPaccuracyScoreRBF)

    elif selectModel == "Decision Tree":

        st.title("Decision Tree Stress Level Estimation Model")
        dtClassifier = DecisionTreeClassifier()

        st.write("Training the model...")
        dtClassifier.fit(data_input_training1, data_target_training)

        st.write("Successfully Train the Model")

        resultsDt = dtClassifier.predict(data_input_testing1)

        st.write("Predicted Results for Testing Dataset:")
        st.write(resultsDt)

        accuracyScoreDt = accuracy_score(resultsDt, data_target_testing)
        st.write("Accuracy Score(Decision Tree):", accuracyScoreDt)

    elif selectModel == "K-Nearest Neighbors":

        st.title("K-Nearest Neighbors Stress Level Estimation Model")
        st.subheader("Number of Nearest Neighbors = 5")

        knn5 = KNeighborsClassifier(n_neighbors=5)

        st.write("Training the model...")
        knn5.fit(data_input_training1, data_target_training)

        st.write("Successfully Train the Model")

        resultsKnn5 = knn5.predict(data_input_testing1)

        st.write("Predicted Results for Testing Dataset:")
        st.write(resultsKnn5)

        PPaccuracyScoreKnn5 = accuracy_score(resultsKnn5, data_target_testing)
        st.write("The Accuracy Score(KNN=5):", PPaccuracyScoreKnn5)

        st.subheader("Number of Nearest Neighbors = 10")

        knn10 = KNeighborsClassifier(n_neighbors=10)

        st.write("Training the model...")
        knn10.fit(data_input_training1, data_target_training)

        st.write("Successfully Train the Model")

        resultsKnn10 = knn10.predict(data_input_testing1)

        st.write("Predicted Results for Testing Dataset:")
        st.write(resultsKnn10)

        PPaccuracyScoreKnn10 = accuracy_score(resultsKnn10, data_target_testing)
        st.write("The Accuracy Score(KNN=10):", PPaccuracyScoreKnn10)

        st.subheader("Number of Nearest Neighbors = 15")

        knn15 = KNeighborsClassifier(n_neighbors=15)

        st.write("Training the model...")
        knn15.fit(data_input_training1, data_target_training)

        st.write("Successfully Train the Model")

        resultsKnn15 = knn15.predict(data_input_testing1)

        st.write("Predicted Results for Testing Dataset:")
        st.write(resultsKnn15)

        PPaccuracyScoreKnn15 = accuracy_score(resultsKnn15, data_target_testing)
        st.write("The Accuracy Score(KNN=15):", PPaccuracyScoreKnn15)

        st.subheader("Number of Nearest Neighbors = 20")

        knn20 = KNeighborsClassifier(n_neighbors=20)

        st.write("Training the model...")
        knn20.fit(data_input_training1, data_target_training)

        st.write("Successfully Train the Model")

        resultsKnn20 = knn20.predict(data_input_testing1)

        st.write("Predicted Results for Testing Dataset:")
        st.write(resultsKnn20)

        PPaccuracyScoreKnn20 = accuracy_score(resultsKnn20, data_target_testing)
        st.write("The Accuracy Score(KNN=20):", PPaccuracyScoreKnn20)

    displayResults = st.sidebar.selectbox("Display the Results", options=["Select Results", "Support Vector Machine", "Decision Tree", "K-Nearest Neighbors"])

    if displayResults == "Support Vector Machine":
            
        as_SVM_Panic = {'Kernel' : ['Linear', 'Polynomial', 'Sigmoid', 'RBF'], 'Accuracy Score' : [PPaccuracyScoreLinear, PPaccuracyScorePoly, PPaccuracyScoreSigmoid, PPaccuracyScoreRBF] }
        tab_SVM_Panic = pd.DataFrame(as_SVM_Panic)
            
        st.subheader('Table of Accuracy Score Results of SVM Models')
        st.write(tab_SVM_Panic)
        st.write('Based on the table above, Linear kernel produces the highest accuracy score which is 0.9583 (95.83%) compare to other kernel. Thus, Linear kernel is chosen to represent SVM model in comparison with other algorithm.')
        
    elif displayResults == "Decision Tree":

        as_DT_Panic = {'Model': ['Decision Tree'], 'Accuracy Score': [accuracyScoreDt]}
        tab_DT_Panic = pd.DataFrame(as_DT_Panic)

        st.subheader('Table of Accuracy Score of Decision Tree Model')
        st.write(tab_DT_Panic)

    elif displayResults == "K-Nearest Neighbors":

        as_KNN_Panic = {'Number of Nearest Neighbors' : ['5','10', '15', '20'],'Accuracy Score': [accuracyScoreKnn5, accuracyScoreKnn10, accuracyScoreKnn15, accuracyScoreKnn20]}
        tab_KNN_Panic = pd.DataFrame(as_KNN_Panic)

        st.subheader('Table of Accuracy Score Results of KNN Models')
        st.write(tab_KNN_Panic)
        st.write('Based on the table above, the Number of Nearest Neighbors 5 produces the same score of highest acuracy score which is 0.9656 (96.56%). Thus, the Number of Nearest Neighbors 5 is chosen to represent KNN model in comparison with other algorithm.')

    conclusion = st.sidebar.selectbox("Conclusion", options=["Select conclusion", "Comparison of Three Models"])

    if conclusion == "Comparison of Three Models":

        st.subheader('Comparison of Results for SVM, KNN, and Decision Tree Models')
        st.write('Comparison of SVM, Decision Tree, and KNN for the Best Model of Panic Disorder Dataset')

        results_models = {'Model': ['SVM (kernel linear)', 'KNN (number of nearest neighbor = 5)', 'Decision Tree'],'Accuracy Score': ['0.9583', '0.9656', '0.9972']}
        tab_result = pd.DataFrame(results_models)
            
        st.write('Table of Comparison')
        st.write(tab_result)
        st.write('Based on the Table above, Decision Tree model produces the best result of highest accuracy score in diagnostic on panic disorder, while the SVM is the worst model. Thus, Decision Tree is selected as the Best Model in diagnostic on panic disorder in this project.')


		