#----------------------------------------------#
#          TOOLBOX FOR BANKS DEFAULT           #
#----------------------------------------------#

# Libraries
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

# //////////////////////////////////////////// #
#--------------- Scatter plot -----------------#

def scatter_plot(df_train, variables, target, x_indx, y_indx):
    X_train = df_train[variables].values # Array of features
    y_train = df_train[target].values # Array of target 
    X_labels = df_train[variables].columns # feature names
    y_label = df_train[target].name # target
    num_plots = len(x_indx)

    if num_plots % 2 == 0:
        f, axs = plt.subplots(num_plots // 2, 2)
    else:
        f, axs = plt.subplots(num_plots// 2 + 1, 2)
      
    f.subplots_adjust(hspace=.3)
    f.set_figheight(10.0)
    f.set_figwidth(10.0)
    
    for i in range(num_plots):
        if i % 2 == 0:
            x_idx = i // 2
            y_idx = 0
        else:
            x_idx = i // 2
            y_idx = 1
          
        axs[x_idx,y_idx].plot(X_train[y_train == 1, x_indx[i]], 
                              X_train[y_train == 1, y_indx[i]], 'rx', label="Default")
        axs[x_idx,y_idx].plot(X_train[y_train == 0, x_indx[i]], 
                              X_train[y_train == 0, y_indx[i]], 'b+',label="No default") 
      
        axs[x_idx,y_idx].legend()
        axs[x_idx,y_idx].set_xlabel('%s' % X_labels[x_indx[i]])
        axs[x_idx,y_idx].set_ylabel('%s' % X_labels[y_indx[i]])
        axs[x_idx,y_idx].set_title('Default vs no default')
        axs[x_idx,y_idx].grid(True)
      
    if num_plots % 2 != 0:
        f.delaxes(axs[i // 2, 1])


# //////////////////////////////////////////// #  
#------------ Calculate metrics ---------------#

def calc_metrics(model, df_test, y_true, threshold=0.5):
    if model is None:
        return 0., 0., 0.
    
    # prediction 
    predicted_sm = model.predict(df_test, linear=False)
    predicted_binary = (predicted_sm > threshold).astype(int)

    # print(predicted_sm.shape, y_true.shape)
    fpr, tpr, _ = metrics.roc_curve(y_true, predicted_sm, pos_label=1)
    
    # compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
    roc_auc = metrics.auc(fpr, tpr)
    ks = np.max(tpr - fpr) # Kolmogorov - Smirnov test

    # note that here teY[:,0] is the same as df_test.default_within_1Y
    accuracy_score = metrics.accuracy_score(y_true, predicted_binary)
    
    try:
        plt.title('Logistic Regression ROC curve')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1], [0,1], 'r--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        # plt.savefig('ROC_curve_1.png')
        plt.show()
    except:
        pass

    return roc_auc, accuracy_score
        
        
# //////////////////////////////////////////// #
#----------------- plot ROC ------------------#

def ROC_plot(y_test, y_pred, model):
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    print('AUC score: %f ' % metrics.roc_auc_score(y_test, y_pred))

    plt.title('%s ROC curve ' % model.__class__.__name__)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.show()
    

    