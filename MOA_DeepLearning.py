import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import tensorflow as tf
import tensorflow_addons as tfa
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.models import load_model
import keras.optimizers as keras_opt
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

warnings.filterwarnings('ignore')

PATH = 'D:/DataScience/4thSem/Data Science Practicum/MOA Dataset Project/'
data = pd.read_csv(PATH+'train_features.csv')
targets = pd.read_csv(PATH+'train_targets_scored.csv')

# Drop cp_type and sig_id
''' cp_type was dropped on the winning notebook of the kaggle competition 
and sig_id is just the id of the patient whose signatures were recorded '''
data = data.drop('cp_type', axis = 1)
targets = targets.drop('sig_id', axis = 1)
data = data.drop('sig_id', axis = 1)
# Encoding Categorical features
data.loc[:,'cp_time'] = data.loc[:, 'cp_time'].map({24: 0, 48: 1, 72: 2})
data['cp_dose'] = data['cp_dose'].apply(lambda x: 0 if x == 'D1' else 1)

# Visualizing the number of observations in each category of the categorical variables
sns.set_theme(style="darkgrid")
fig, ax = plt.subplots()
ax = sns.countplot(x = 'cp_time', data = data)
ax.set_title('Observations in each category of the feature cp_time')
plt.show()
fig.savefig(PATH+'cp_time.jpg', dpi = 600)

fig, ax = plt.subplots()
ax = sns.countplot(x = 'cp_dose', data = data)
ax.set_title('Observations in each category of the feature cp_dose')
plt.show()
fig.savefig(PATH+'cp_dose.jpg', dpi = 600)

# Separating the GENE and CELL column names for later use in PCA
feature_cols = data.columns.tolist()
target_cols = targets.columns.tolist()
GENES = [col for col in data.columns if col.startswith('g-')]
CELLS = [col for col in data.columns if col.startswith('c-')]

# Quantile Transformation 
''' Machine learning models works the best with Gaussian distributions.
So we transformed the dist of the data. '''
qt = QuantileTransformer(n_quantiles = 100, output_distribution = 'normal')
data2 = qt.fit_transform(data[GENES+CELLS])
data2 = pd.DataFrame(data2, columns=(GENES+CELLS))  

X_train, X_test, y_train, y_test = train_test_split(data2, targets, test_size = 0.2)

# Visualising one transformed vector each from the cell and gene data 
# 'data2' has the transformed vectors
fig,ax = plt.subplots()
sns.distplot(data['g-10'], bins = 20, rug = True)
fig.savefig(PATH+'g-10.jpg', dpi = 600)

fig,ax = plt.subplots()
sns.distplot(data2['g-10'], bins = 20, rug = True)
fig.savefig(PATH+'g-10-QT.jpg', dpi = 600)

fig,ax = plt.subplots()
sns.distplot(data['c-5'], bins = 20, rug = True)
fig.savefig(PATH+'c-5.jpg', dpi = 600)

fig,ax = plt.subplots()
sns.distplot(data2['c-5'], bins = 20, rug = True)
fig.savefig(PATH+'c-5-QT.jpg', dpi = 600)


# Principal Component Analysis
''' We did PCA separately for GENE and CELL data because both of them
    represent different information and we don't want to mingle those up.'''

""" PCA was done twice. First with manually set feature space of 67.
    50 for gene vectors and 15 for cell vectors. 2 categorical features. 
    Varince explained was about 70% for both GENE and CELL vectors. 
    
    Next the algorithm was allowed to choose number of PCs to preserve 95% variacne.
    Feature space came out to be 682.
"""
pca_genes = PCA(n_components = 0.95).fit(X_train[GENES])
pca_cells = PCA(n_components = 0.95).fit(X_train[CELLS])

data_genes_transformed = pca_genes.transform(data2[GENES])
data_cells_transformed = pca_cells.transform(data2[CELLS])

data3 = pd.concat((data[['cp_time', 'cp_dose']], 
                   pd.DataFrame(data_genes_transformed), 
                  pd.DataFrame(data_cells_transformed)), axis = 1)


X_train, X_test, y_train, y_test = train_test_split(data3, targets, test_size = 0.2)

X_train = X_train.reset_index(drop = True)
y_train = y_train.reset_index(drop = True)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Renaming the principal component columns of the gene and cell data
col_names = ['cp_time', 'cp_dose']+['g_PC'+str(i) for i in range(1,601)]+ \
['c_PC'+str(i) for i in range(601,684)]
X_train.columns = col_names
X_test.columns = col_names

'''
# Sklearn MLP - We started with this package but initial testing showed us 
    that Keras was performing better and was more customizable. 
    So, this section was scrapped. 
    
NSeeds = 3
NFolds = 5
def get_model():    
    MLP = MLPClassifier(hidden_layer_sizes = 1024, max_iter = 1000, 
                        learning_rate = 'adaptive',
                        batch_size = 1024, solver = 'adam', tol = 1e-4, verbose=True)
    return MLP

def run_kfold(train, targets, NFolds, seed):
    sum_acc = 0
    mskf = MultilabelStratifiedKFold(n_splits = NFolds)
    for f, (trn_idx, val_idx) in enumerate(mskf.split(train, targets)):
        X_train, X_val = train.loc[trn_idx,:], train.loc[val_idx,:]
        y_train, y_val = targets.loc[trn_idx,:], targets.loc[val_idx,:]
        
        model = get_model()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        sum_acc += acc
        print("Seed {} Fold {} : Val_accuracy = {}".format(seed, f, acc))
    
    return sum_acc/NFolds
        
        
def run_seeds(train, targets, nfolds = NFolds, nseeds = NSeeds):
    acc = []
    for seed in range(nseeds):
        acc.append(run_kfold(train, targets, nfolds, seed))
    print("Average Validation Accuracy = {}".format(sum(acc)/nseeds))
        

run_seeds(X_train, y_train)
mlp = get_model()

mlp.fit(X_train, y_train)

y_pred_mlp = mlp.predict(X_test)
y_pred_train_mlp = mlp.predict(X_train)

accuracy_score(y_train, y_pred_train_mlp)
accuracy_score(y_test, y_pred_mlp)
'''

#sparsemax = tfa.layers.Sparsemax()
#sparsemax_loss = tfa.losses.sparsemax_loss()
# Keras Sequential - model definition
def NN_Model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(1024, input_dim = n_inputs, kernel_initializer='he_uniform', 
                    kernel_regularizer=l2(0.01), activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dropout(0.4))
    #model.add(Dense(64, activation = 'relu'))
    model.add(Dense(n_outputs, activation = 'sigmoid'))
    opt = keras_opt.Nadam(lr = 0.001)
    model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics=['accuracy'])
    return model

n_inputs, n_outputs = X_train.shape[1], y_train.shape[1]

# Saving the training time of all the models
# We also save the models for later usage 
training_time = []

# Model 1 training 
start = time.time()
nn_1 = NN_Model(n_inputs, n_outputs)
nn_1.fit(X_train, y_train, batch_size = 32, epochs=200)
training_time.append(time.time()-start)
nn_1.save(PATH+'Models_v2/Model_1')

# Model 2 training
start = time.time()
nn_2 = NN_Model(n_inputs, n_outputs)
nn_2.fit(X_train, y_train, batch_size = 1024, epochs=1000)
training_time.append(time.time()-start)
nn_2.save(PATH+'Models_v2/Model_2')

# Saving the prediction time of all the models
prediction_time = []

# Model 1 predictions
start = time.time()
y_pred = pd.DataFrame(nn_1.predict(X_test))
y_pred_train = pd.DataFrame(nn_1.predict(X_train))
prediction_time.append(time.time()-start)
# Model 2 predictions
start = time.time()
y_pred2 = pd.DataFrame(nn_2.predict(X_test))
y_pred_train2 = pd.DataFrame(nn_2.predict(X_train))
prediction_time.append(time.time()-start)

# Accuracy score was chosen as the initial metric but then precision and recall 
# were adopted because those were more important for the business problem. 
accuracy_score(y_train, y_pred_train2.round())
accuracy_score(y_test, y_pred2.round())
precision_score(y_train, y_pred_train2.round(), average = 'micro')
precision_score(y_test, y_pred2.round(), average = 'micro')

# ROC curve analysis - Checking if the models were predicting anything at all
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(206):
    fpr[i], tpr[i], _ = roc_curve(y_test.iloc[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i] )
    
fpr_trn = dict()
tpr_trn = dict()
roc_auc_trn = dict()
for i in range(206):
    fpr_trn[i], tpr_trn[i], _ = roc_curve(y_train.iloc[:, i], y_pred_train[:, i])
    roc_auc_trn[i] = auc(fpr_trn[i], tpr_trn[i] )

plt.figure()
lw = 2
plt.plot(fpr[105], tpr[105], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[105])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()



# P-R curve analysis
prec = dict()
rec = dict()
pr_auc = dict()
for i in range(206):
    prec[i], rec[i], _ = precision_recall_curve(y_test.iloc[:, i], y_pred[:, i])
    pr_auc[i] = auc(rec[i], prec[i])
    
prec_trn = dict()
rec_trn = dict()
pr_auc_trn = dict()
for i in range(206):
    prec_trn[i], rec_trn[i], _ = precision_recall_curve(y_train.iloc[:, i], y_pred_train[:, i])
    pr_auc_trn[i] = auc(rec_trn[i], prec_trn[i])

plt.figure()
lw = 2
plt.plot(rec[35], prec[35], color='darkorange',
         lw=lw, label='PR curve (area = %0.2f)' % pr_auc[35])
plt.hlines(0.05, 0, 1, color='navy', lw=lw, linestyles='dashed', label = 'No Skill')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve')
plt.legend(loc="upper right")
plt.show()

# Testing probability thresholding
''' We needed to achieve 90% precision, so this method was used to get the threshold 
    at which the precision reached the desired value.'''

thresholds = np.arange(0, 1, 0.05)
# Function to convert class probabilities to class labels
def to_labels(thresh, y_pred):
    y_pred_labels = y_pred.applymap(lambda x: 1 if x >= thresh else 0)
    return y_pred_labels
# Precision at different thresholds for model 1 and model 2
prec_scores_train = [precision_score(y_train, to_labels(t, y_pred_train), average='micro') for t in thresholds]
prec_scores_test = [precision_score(y_test, to_labels(t, y_pred), average='micro') for t in thresholds]
prec_scores_train2 = [precision_score(y_train, to_labels(t, y_pred_train2), average='micro') for t in thresholds]
prec_scores_test2 = [precision_score(y_test, to_labels(t, y_pred2), average='micro') for t in thresholds]

# Recall at different thresholds for model 1 and model 2
rec_scores_train = [recall_score(y_train, to_labels(t, y_pred_train), average='micro') for t in thresholds]
rec_scores_test = [recall_score(y_test, to_labels(t, y_pred), average='micro') for t in thresholds]


# Effect on precision and recall while varying the probability threshold for class labelling  
fig, ax = plt.subplots()
plt.plot(thresholds, prec_scores_test, color = "red")
plt.plot(thresholds, rec_scores_test, color = "blue")
plt.xlabel("Threshold")
plt.ylabel("Precision Score | Recall Score")
plt.title("Precision & Recall at varying Thresholds - Test Set")  
plt.legend(["Precision", "Recall"], loc = "center right")   
fig.savefig(PATH+"Prec_Rec_Test_Threshold.jpg", dpi = 600)

fig, ax = plt.subplots()
plt.plot(thresholds, prec_scores_train, color = "red")
plt.plot(thresholds, rec_scores_train, color = "blue")
plt.xlabel("Threshold")
plt.ylabel("Precision Score | Recall Score")  
plt.title("Precision & Recall at varying Thresholds - Training Set")
plt.legend(["Precision", "Recall"], loc = "center right")   
fig.savefig(PATH+"Prec_Rec_Train_Threshold.jpg", dpi = 600)

# Find top 5 and bottom five labels
# Analysis of results of probability thresholding
support = [targets[col].value_counts()[1] for col in target_cols]
n = len(support)
top5 = sorted(range(n), key=lambda i: support[i], reverse=True)[:5]
bot5 = sorted(range(n), key=lambda i: support[i])[:5]

# All, Top 100, minority support ratio and the labels
support_ratio = [x/data.shape[0] for x in support]
temp = sorted(support_ratio, reverse=True)
all_labels_sorted = sorted(range(n), key=lambda i: support[i], reverse=True)
all_labels_sorted_supRatio = pd.concat([pd.Series(all_labels_sorted), pd.Series(temp)], axis = 1)
top100 = sorted(range(n), key=lambda i: support[i], reverse=True)[:100]
top100_supRatio = pd.concat([pd.Series(top100), pd.Series(temp[:100])], axis = 1)
bot50 = sorted(range(n), key=lambda i: support[i], reverse=True)[51:100]
bot50_supRatio = pd.concat([pd.Series(bot50), pd.Series(temp[51:100])], axis = 1)


# Custom function to get precision and recall for specified labels
def get_precision(labels, y_test, y_pred):
    prec_scores = []
    for label in labels:
        prec_scores.append(precision_score(y_test.iloc[:,label], y_pred.iloc[:,label]))
    return prec_scores

def get_recall(labels, y_test, y_pred):
    rec_scores = []
    for label in labels:
        rec_scores.append(recall_score(y_test.iloc[:,label], y_pred.iloc[:,label]))
    return rec_scores


# Precision and Recall values at varying threshold for the top5 labels
prec_top5 = []
for t in thresholds:
    y_pred_labels = to_labels(t, y_pred)
    prec_top5.append(get_precision(top5, y_test, y_pred_labels))
    
prec_top5 = pd.DataFrame(prec_top5)

rec_top5 = []
for t in thresholds:
    y_pred_labels = to_labels(t, y_pred)
    rec_top5.append(get_recall(top5, y_test, y_pred_labels))

# Visualizing the results  
fig, ax = plt.subplots()
plt.plot(thresholds, prec_top5.iloc[:,0], color = "red")
plt.plot(thresholds, prec_top5.iloc[:,1], color = "blue")
plt.plot(thresholds, prec_top5.iloc[:,2], color = "green")
plt.plot(thresholds, prec_top5.iloc[:,3], color = "black")
plt.plot(thresholds, prec_top5.iloc[:,4], color = "orange")
plt.xlabel("Threshold")
plt.ylabel("Precision Score") 
plt.title("Test Set - Model1") 
plt.legend(["Label - 136", "Label - 163", "Label - 71", "Label - 79", "Label - 177"], loc = "center right")   
fig.savefig(PATH+"Threshold_32_test.jpg", dpi = 600)


''' Detailed Recall Analysis of Model 1 and Model 2'''

' ----- Model 1 ----- '
# Recall values of labels at 90% precision
''' Test set - Precision reaches 90% at 0.5 '''
recall_values = get_recall(range(0,206), y_test, to_labels(0.6, y_pred))
# Labels having recall > 0
has_recall = [{recall_values.index(x):x} for x in recall_values if x > 0]

top100_recall = [recall_values[i] for i in top100]

#temp2 = [i for i in top100_recall if i>0]

top100_supRatio_recall = pd.concat([top100_supRatio, pd.Series(top100_recall)], axis = 1)
top100_supRatio_recall.columns = ["Label", "Support Ratio", "Recall Score"]

''' Train set - Precision reaches 90% at 0.45 '''
recall_values_train = get_recall(range(0,206), y_train, to_labels(0.55, y_pred_train))
# Labels having recall > 0
has_recall_train = [{recall_values_train.index(x):x} for x in recall_values_train if x > 0]

top100_recall_train = [recall_values_train[i] for i in top100]

#temp3 = [i for i in top100_recall_train if i>0]

# Putting together the results of top 100 labels
top100_supRatio_recall_train = pd.concat([top100_supRatio, pd.Series(top100_recall_train)], axis = 1)
top100_supRatio_recall_train.columns = ["Label", "Support Ratio", "Recall Score"]


'----- Model 2 -----'
# Recall values of labels at 90% precision
''' Test set - Precision reaches 90% at 0.75 '''
recall_values2 = get_recall(range(0,206), y_test, to_labels(0.7, y_pred2))
recall_values_sorted = [recall_values2[i] for i in all_labels_sorted]
# Labels having recall > 0
has_recall2 = [{recall_values2.index(x):x} for x in recall_values2 if x > 0]

top100_recall2 = [recall_values2[i] for i in top100]

#temp4 = [i for i in top100_recall2 if i>0]

top100_supRatio_recall2 = pd.concat([top100_supRatio, pd.Series(top100_recall2)], axis = 1) 
top100_supRatio_recall2.columns = ["Label", "Support Ratio", "Recall Score"]
all_supRatio_recall2 = pd.concat([all_labels_sorted_supRatio, pd.Series(recall_values_sorted)], axis = 1)
all_supRatio_recall2.columns = ["Label", "Support Ratio", "Recall Score"]

''' Train set - Precision reaches 90% at 0.3 '''
recall_values_train2 = get_recall(range(0,206), y_train, to_labels(0.2, y_pred_train2))
recall_values_sorted_train = [recall_values_train2[i] for i in all_labels_sorted]
# Labels having recall > 0
has_recall_train2 = [{recall_values_train2.index(x):x} for x in recall_values_train2 if x > 0]

top100_recall_train2 = [recall_values_train2[i] for i in top100]

#temp5 = [i for i in top100_recall_train2 if i>0]

# Putting together the results of top 100 labels
top100_supRatio_recall_train2 = pd.concat([top100_supRatio, pd.Series(top100_recall_train2)], axis = 1)
top100_supRatio_recall_train2.columns = ["Label", "Support Ratio", "Recall Score"]
# Putting together the results of all labels
all_supRatio_recall_train2 = pd.concat([all_labels_sorted_supRatio, pd.Series(recall_values_sorted_train)], axis = 1)
all_supRatio_recall_train2.columns = ["Label", "Support Ratio", "Recall Score"]

''' Global, Majority and minority recall analysis '''
# Global recall of 206 labels
global_recall_train = sum([x*y for x,y in zip(all_supRatio_recall_train2['Support Ratio'], 
                                              all_supRatio_recall_train2['Recall Score'])])

global_recall_test = sum([x*y for x,y in zip(all_supRatio_recall2['Support Ratio'], 
                                              all_supRatio_recall2['Recall Score'])])

# Recall of top 100 labels with scaled support ratio 
sum_100 = sum(top100_supRatio.iloc[:,1])
top100_supRatio_scaled = [x/sum_100 for x in top100_supRatio.iloc[:,1]]
top100_totalRecall_train = sum(x*y for x, y in zip(top100_supRatio_scaled, 
                                              top100_recall_train2)) 

top100_totalRecall_test = sum(x*y for x, y in zip(top100_supRatio_scaled, 
                                              top100_recall2)) 
  
# Recall of minority labels (51 - 100) with scaled support ratio  
sum_50 = sum(top100_supRatio.iloc[51:101,1])
bot50_supRatio_scaled = [x/sum_50 for x in top100_supRatio.iloc[51:101,1]]
minority_recall_train = sum([x*y for x,y in zip(bot50_supRatio_scaled, 
                                              top100_recall_train2[51:101])])

minority_recall_test = sum([x*y for x,y in zip(bot50_supRatio_scaled, 
                                              top100_recall2[51:101])])


''' ALL OF THE STEPS BELOW INCLUDING MODEL TRAINING, PRECISION THRESHOLDING, 
    RECALL ANALYSIS AND VISUALIZATION WERE DONE TWICE. ONCE WITH FEATURE SPACE OF 
    67 AND THEN WITH FEATURE SPACE OF 682. THE CODE REFLECTS THE LATER ONLY. '''

# Recall analysis on more models
''' 
    First we need to find the probability thresholds for all the models when 
    precision reaches 90%. We define a function for that purpose, then train 
    the models with their respective hyperparameters and loop over the function 
    to get the precision values at varied thresholds.
'''

def get_precision_thresholding(y_train, y_pred_train, y_test, y_pred, thresholds):
    prec_scores_train = [precision_score(y_train, to_labels(t, y_pred_train), average='micro') for t in thresholds]
    prec_scores_test = [precision_score(y_test, to_labels(t, y_pred), average='micro') for t in thresholds]
    return [prec_scores_train, prec_scores_test]

# We would need to change this definition everytime we need to train a new model
def NN_Model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(1024, input_dim = n_inputs, kernel_initializer='he_uniform', 
                    kernel_regularizer=l2(0.0001), activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(n_outputs, activation = 'sigmoid'))
    opt = keras_opt.Nadam(lr = 0.001)
    model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics=['accuracy'])
    return model

# We already have model 1 and model 2. We will train 8 more models to compare.
# Model 3 training 
start = time.time()
nn_3 = NN_Model(n_inputs, n_outputs)
nn_3.fit(X_train, y_train, batch_size = 1024, epochs=1000)
training_time.append(time.time()-start)
nn_3.save(PATH+'Models_v2/Model_3')    

# Model 4 training 
start = time.time()
nn_4 = NN_Model(n_inputs, n_outputs)
nn_4.fit(X_train, y_train, batch_size = 256, epochs=400)
training_time.append(time.time()-start)
nn_4.save(PATH+'Models_v2/Model_4')

# Model 5 training 
start = time.time()
nn_5 = NN_Model(n_inputs, n_outputs)
nn_5.fit(X_train, y_train, batch_size = 256, epochs=2000)
training_time.append(time.time()-start)
nn_5.save(PATH+'Models_v2/Model_5')

# Model 6 training 
start = time.time()
nn_6 = NN_Model(n_inputs, n_outputs)
nn_6.fit(X_train, y_train, batch_size = 1024, epochs=2000)
training_time.append(time.time()-start)
nn_6.save(PATH+'Models_v2/Model_6')

# Model 7 training
start = time.time() 
nn_7 = NN_Model(n_inputs, n_outputs)
nn_7.fit(X_train, y_train, batch_size = 256, epochs=200)
training_time.append(time.time()-start)
nn_7.save(PATH+'Models_v2/Model_7')

# Model 8 training 
start = time.time()
nn_8 = NN_Model(n_inputs, n_outputs)
nn_8.fit(X_train, y_train, batch_size = 1024, epochs=1000)
training_time.append(time.time()-start)
nn_8.save(PATH+'Models_v2/Model_8')

# Model 9 training 
start = time.time()
nn_9 = NN_Model(n_inputs, n_outputs)
nn_9.fit(X_train, y_train, batch_size = 512, epochs=1000)
training_time.append(time.time()-start)
nn_9.save(PATH+'Models_v2/Model_9')

# Model 10 training 
start = time.time()
nn_10 = NN_Model(n_inputs, n_outputs)
nn_10.fit(X_train, y_train, batch_size = 1024, epochs=2000)
training_time.append(time.time()-start)
nn_10.save(PATH+'Models_v2/Model_10')

traning_time_min = [x/60 for x in training_time]

""" Loading saved models for reusage is pretty easy.
nn_1 = load_model(PATH+'Models/Model_1')
nn_2 = load_model(PATH+'Models/Model_2')
nn_3 = load_model(PATH+'Models/Model_3')
nn_4 = load_model(PATH+'Models/Model_4')
nn_5 = load_model(PATH+'Models/Model_5')
nn_6 = load_model(PATH+'Models/Model_6')
nn_7 = load_model(PATH+'Models/Model_7')
nn_8 = load_model(PATH+'Models/Model_8')
nn_9 = load_model(PATH+'Models/Model_9')
nn_10 = load_model(PATH+'Models/Model_10')
"""

# Saving the models in a list for looping over them
models = [nn_1, nn_2, nn_3, nn_4, nn_5, nn_6, nn_7, nn_8, nn_9, nn_10]

# Saving all the predictions and precision values 
model_preds_train = []
model_preds_test = []
prec_train_all_models = []
prec_test_all_models = []

# Getting precision for varied probability thresholds for additional models 
for i, model in enumerate(models):
    print("--- Executing Model", i+1, "predictions ---", "\n")
    # Recording prediction times
    start = time.time()
    y_pred_train = pd.DataFrame(model.predict(X_train))
    y_pred = pd.DataFrame(model.predict(X_test))
    prediction_time.append(time.time()-start)
    # Recording the predictions
    model_preds_train.append(y_pred_train)
    model_preds_test.append(y_pred)
    # Getting precision values at varied thresholds
    temp = get_precision_thresholding(y_train, y_pred_train, y_test, y_pred, thresholds)
    prec_train_all_models.append(temp[0])
    prec_test_all_models.append(temp[1])
    
# Probability thresholds when each model reaches a 90% precision value.
# The values were visually and manually determined from the above results.    
thresh_90percent_train = [0.55, 0.2, 0.35, 0.35, 0.35, 0.4, 0.3, 0.1, 0.15, 0.1]
thresh_90percent_test = [0.6, 0.7, 0.7, 0.65, 0.65, 0.65, 0.9, 0.9, 0.9, 0.95]

# Custom function to get global, major and minor recall values for a model
# This function expects the labels sorted according to their support ratio.
def recall_analysis(y_pred, y_test, thresh, all_labels_sorted, all_labels_sorted_supRatio):
    '''Calculate the recall values for all the labels individually and sort them
    according to their support ratio '''
    recall_values = get_recall(range(0,206), y_test, to_labels(thresh, y_pred))
    recall_values_sorted = [recall_values[i] for i in all_labels_sorted]
    
    # This is a dataframe where the columns are in order: Label, support ratio, recall value
    all_supRatio_recall = pd.concat([all_labels_sorted_supRatio, 
                                    pd.Series(recall_values_sorted)], axis = 1)
    
    # Global recall. sum(Sup ratio * recall) for all labels
    global_recall = sum([x*y for x, y in zip(all_supRatio_recall.iloc[:,1], 
                                            all_supRatio_recall.iloc[:,2])])
    
    # Scale support ratio of top 100 labels and calculate majority recall
    sum_100 = sum(all_supRatio_recall.iloc[0:100,1])
    top_100_scaled = [x/sum_100 for x in all_supRatio_recall.iloc[0:100,1]]
    majority_recall = sum([x*y for x, y in zip(top_100_scaled,
                                              all_supRatio_recall.iloc[0:100,2])])
    
    # Scale support ratio of labels 51-100 and calculate minority recall
    sum_50 = sum(all_supRatio_recall.iloc[51:101,1])
    bot_50_scaled = [x/sum_50 for x in all_supRatio_recall.iloc[51:101,1]]
    minority_recall = sum([x*y for x, y in zip(bot_50_scaled, 
                                               all_supRatio_recall.iloc[51:100,2])])
    
    return [global_recall, majority_recall, minority_recall]

# Traning set results for 10 models 
results_recall_train = []
i = 0
for thresh, y_pred in zip(thresh_90percent_train, model_preds_train):
    print("--- Analysing Model", i+1, ": Getting recall values ---", "\n")
    res = recall_analysis(y_pred, y_train, thresh, all_labels_sorted, all_labels_sorted_supRatio)
    results_recall_train.append(res)
    i += 1

# Test set results for 10 models
results_recall_test = []
i = 0
for thresh, y_pred in zip(thresh_90percent_test, model_preds_test):
    print("--- Analysing Model", i+1, ": Getting recall values ---", "\n")
    res = recall_analysis(y_pred, y_test, thresh, all_labels_sorted, all_labels_sorted_supRatio)
    results_recall_test.append(res)
    i += 1

# Organizing the results    
indexes = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 
           'Model 6', 'Model 7', 'Model 8', 'Model 9', 'Model 10']
columns = ['Global Recall', 'Majority Recall', 'Minority Recall']

results_recall_train = pd.DataFrame(results_recall_train, index=indexes, columns=columns)
results_recall_test = pd.DataFrame(results_recall_test, index=indexes, columns=columns)

# Visualizing the results obtained
# Training set results
plt.style.use('ggplot')
fig, ax = plt.subplots()
ax.plot(results_recall_train.index, results_recall_train['Global Recall'], 
        label = 'Global Recall', color = 'b')
ax.plot(results_recall_train.index, results_recall_train['Majority Recall'], 
        label = 'Majority Recall', color = 'g')
ax.plot(results_recall_train.index, results_recall_train['Minority Recall'], 
        label = 'Minority Recall', color = 'r')
ax.set_xlabel('Models')
ax.set_ylabel('Recall Values')
ax.set_title('Training set recall analysis')
plt.xticks(rotation=90)
plt.legend()
plt.show()
fig.set_size_inches([8,4])
fig.savefig(PATH+'Models_v2/Recall_Analysis_Training.jpg', dpi = 600, bbox_inches = "tight")

# Test set results
fig2, ax2 = plt.subplots()
ax2.plot(results_recall_test.index, results_recall_test['Global Recall'], 
        label = 'Global Recall', color = 'b')
ax2.plot(results_recall_test.index, results_recall_test['Majority Recall'], 
        label = 'Majority Recall', color = 'g')
ax2.plot(results_recall_test.index, results_recall_test['Minority Recall'], 
        label = 'Minority Recall', color = 'r')
ax2.set_xlabel('Models')
ax2.set_ylabel('Recall Values')
ax2.set_title('Test set recall analysis')
plt.xticks(rotation=90)
plt.legend(loc = 'lower right')
plt.show()
fig2.set_size_inches([8,4])
fig2.savefig(PATH+'Models_v2/Recall_Analysis_Test.jpg', dpi = 600, bbox_inches = "tight")


''' ALL OF THE STEPS ABOVE INCLUDING MODEL TRAINING, PRECISION THRESHOLDING, 
    RECALL ANALYSIS AND VISUALIZATION WERE DONE TWICE. ONCE WITH FEATURE SPACE OF 
    67 AND THEN WITH FEATURE SPACE OF 682. THE CODE REFLECTS THE LATER ONLY. '''

    
# Model re-training
"""  
    We will try to improve the recall score on test sets for the models
    by re-training the saved models with only the bottom 100 (minority) labels 
    w.r.t support ratio. 
"""
""" 
    This section was not completed and can be considered for furthur research
    because retraining the models with minority labels were giving worse results 
    for all the labels.
"""

y_train_minor = y_train.copy(deep=True)
# Setting the top 100 labels to 0 for all observations. We want to retrain the model 
# with only the effects of the minority labels
for label in all_labels_sorted[0:101]:
    y_train_minor.iloc[:,label].values[:] = 0

models = [nn_1, nn_2, nn_3, nn_4, nn_5, nn_6, nn_7, nn_8, nn_9, nn_10]
batch_size = [32, 1024, 1024, 256, 256, 1024, 256, 1024, 512, 1024]
epochs = [200, 1000, 1000, 400, 2000, 2000, 200, 1000, 1000, 2000]
i = 0
for model, batch, epoch in zip(models, batch_size, epochs):
    print("Retraining Model: ", i+1, "\n")
    model.fit(X_train, y_train_minor, batch_size = batch, epochs = epoch, 
              verbose = 0)
    i+=1

model_preds_train2 = []
model_preds_test2 = []
prec_test_all_models2 = []
prec_train_all_models2 = []

for i, model in enumerate(models):
    print("--- Executing Model", i+1, "predictions ---", "\n")
   
    y_pred_train = pd.DataFrame(model.predict(X_train))
    y_pred = pd.DataFrame(model.predict(X_test))
    
    model_preds_train2.append(y_pred_train)
    model_preds_test2.append(y_pred)
  
    temp = get_precision_thresholding(y_train, y_pred_train, y_test, y_pred, thresholds)
    prec_train_all_models2.append(temp[0])
    prec_test_all_models2.append(temp[1])

# precision did not reach 90% for the test set for most models
thresh_train = [0.3, 0.35, 0.35, 0.3, 0.35, 0.25, 0.1, 0.1, 0.1]
thresh_test = []

# ... to be continued








