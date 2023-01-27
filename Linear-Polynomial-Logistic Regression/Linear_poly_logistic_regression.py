import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigm(x):
    return np.exp(-x)/(1 + np.exp(-x))

def score_logistic(Y_test, prediction):
    tp = np.sum(np.logical_and(prediction == Y_test, Y_test == 1))
    tn = np.sum(np.logical_and(prediction == Y_test, Y_test == 0))
    fp = np.sum(np.logical_and(prediction != Y_test, Y_test == 0))
    fn = np.sum(np.logical_and(prediction != Y_test, Y_test == 1))
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    
    return {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'Precision': p,
        'Recall': r,
        'NPV': tn / (tn + fn),
        'FPR': fp / (fp + tn),
        'FDR': fp / (fp + tp),
        
        'F1': (1 + 1 ** 2) * p * r / (1 ** 2 * p + r),
 
        'F2': (1 + 2 ** 2) * p * r / (2 ** 2 * p + r)
    }


    
#normalize the data
def normalize_data(data):
    centralized_data =(data - np.mean(data,axis=0)) / np.std(data,axis=0)
    return centralized_data


def centralize_data(data):
    return (data - np.mean(data,axis=0))





#given the data and labels, calculates linear regression with LMS
def linear_regression(x,y):
    #add the bias column to X
    bias_col = np.ones(len(x)).reshape(-1,1)
    x_bias = np.concatenate((bias_col,x),axis = 1)
    
    #the derived general formula for linear regression
    xtx =np.matmul(np.transpose(x_bias),x_bias)
    inv_xtx = np.linalg.inv(xtx)
    xty = np.matmul(np.transpose(x_bias),y)
    B_weights = np.matmul(inv_xtx,xty)
    
    #predict the prices according to B
    y_prediction = np.matmul(x_bias,B_weights)
    return B_weights, y_prediction 
    

#this function makes a polynomial regression with degree 2
def linear_regression_polynomial(x,y):
    #add bias column and x^2 to the training data
    bias_col = np.ones(len(x)).reshape(-1,1)
    x_2 = np.square(x)
    x_final = np.concatenate([bias_col,x,x_2],axis = 1)
    
    #calculate the regression 
    xtx =np.matmul(np.transpose(x_final),x_final)
    inv_xtx = np.linalg.inv(xtx)
    xty = np.matmul(np.transpose(x_final),y)
    B_weights = np.matmul(inv_xtx,xty)
    
    #ther prdixtion is y_hat = XB
    y_prediction = np.matmul(x_final,B_weights)
    return B_weights,y_prediction
    
    
#this function calculates the MSE given y and predicitions
def MSE(y_hat_fun,y_fun):
    y_hat_2 =y_hat_fun.to_numpy()
    y_2 = y_fun.to_numpy()
    diff = y_2-y_hat_2
    mse = np.mean(np.square(diff))
  
    return mse




#------------------------Q1-----------------------------------------------


images = pd.read_csv("images.csv")
#normalize the data
images = centralize_data(images)

#find the cov matrix
cov_matrix = np.cov(images)
#find eigenvalues and eigen vectors
eigen_values,eigen_vectors = np.linalg.eigh(cov_matrix)


#sort the eigenvalues in descending order
sort_index = np.argsort(eigen_values)[::-1]

#find the sorted eigenvectors and eigenvalues 
#first line will be the highest captured variance
sorted_eig_val = eigen_values[sort_index]
sorted_eig_vec = eigen_vectors[sort_index]

#take the first 10 principal components
eig_val_10 = sorted_eig_val[:10]
eig_vec_10 = sorted_eig_vec[:10]

#find the PVE for first 10 PC
pve_total = 0
total_variance = np.sum(eigen_values)
for k in range(10):
    pve_k = eig_val_10[k]/total_variance
    print('The PVE of ',k,'th priciple component is ',pve_k)
    pve_total = pve_total + pve_k

#find the reduced data
reduced_images = (eig_vec_10 @ images).to_numpy()
 

#tka
plt.figure()
#show the images of the first 10 components
for i in range(2):    
    for l in range(5):
        reduced_img = reduced_images[5*i +l]
        reduced_img = reduced_img.reshape(48,48)
        plt.subplot(2,5,5*i +l+1)
        plt.imshow(reduced_img,interpolation='nearest')





#get the PVE for k = 1,10,50,100,500
 

k_vec = [1,10,50,100,500]
pve_total_list = []
#this loop will work for every k value
for k in k_vec:
    #loop for a specific k value
    pve_total = 0
    #find sum of pve of first k components
    for l in range(k):
        
        pve_k = sorted_eig_val[l]/total_variance
        pve_total = pve_total + pve_k
    #list of the pve of desired k values     
    pve_total_list.append(pve_total)
    print('PVE for first',k, 'principal component is ',pve_total)



plt.figure()
plt.plot(k_vec,pve_total_list)
plt.xlabel("first k components")
plt.ylabel("PVE captured total")

#1.3
#find the eigen vectors for k = 1,10,50,100,500
eig_vec_1 = sorted_eig_vec[:1]
eig_vec_10 = sorted_eig_vec[:10]
eig_vec_50 = sorted_eig_vec[:50]
eig_vec_100 = sorted_eig_vec[:100]
eig_vec_500 = sorted_eig_vec[:500]

#find the priciple components and then the reduced images, take one sample of image and show it with subplots
#k = 1
pc_image = (eig_vec_1 @ images).to_numpy()
reduced_img_k = np.transpose(pc_image) @ eig_vec_1
one_img_sample = np.transpose(reduced_img_k)[0].reshape(48,48)
plt.figure()
plt.subplot(1,5,1)
plt.imshow(one_img_sample,interpolation='nearest')


#k = 10
pc_image = (eig_vec_10 @ images).to_numpy()
reduced_img_k = np.transpose(pc_image) @ eig_vec_10
one_img_sample_10 = np.transpose(reduced_img_k)[0].reshape(48,48)
plt.subplot(1,5,2)
plt.imshow(one_img_sample_10,interpolation='nearest')


#k = 50
pc_image = (eig_vec_50 @ images).to_numpy()
reduced_img_k = np.transpose(pc_image) @ eig_vec_50
one_img_sample_50 = np.transpose(reduced_img_k)[0].reshape(48,48)
plt.subplot(1,5,3)
plt.imshow(one_img_sample_50,interpolation='nearest')


#k = 100
pc_image = (eig_vec_100 @ images).to_numpy()
reduced_img_k = np.transpose(pc_image) @ eig_vec_100
one_img_sample_100 = np.transpose(reduced_img_k)[0].reshape(48,48)
plt.subplot(1,5,4)
plt.imshow(one_img_sample_100,interpolation='nearest')
 
#k = 500   
pc_image = (eig_vec_500 @ images).to_numpy()
reduced_img_k = np.dot(np.transpose(pc_image) ,eig_vec_500)
one_img_sample_500 = np.transpose(reduced_img_k)[0].reshape(48,48)
plt.subplot(1,5,5)
plt.imshow(one_img_sample_500,interpolation = 'nearest')









#--------------------------------Q2----------------------------------------

#get_data and normalize features
data_train = pd.read_csv("question-2-features.csv")
data_train_normalized = normalize_data(data_train)
train_labels = pd.read_csv("question-2-labels.csv")


# find rank XTX
XTX = np.matmul(np.transpose(data_train),data_train)
rank_XTX = np.linalg.matrix_rank(XTX)
print("The rank of XTX is ", rank_XTX)


#Calculate the weights for the last feature of the dataset
X_13 = data_train['LSTAT'].to_numpy().reshape(-1,1)#reshape the data for 2 dimension
B , y_hat = linear_regression(X_13,train_labels) #find the weights and prediciton
B = B.to_numpy()


#plot the data and estimation
plt.figure()
plt.plot(X_13,y_hat,label = 'Linear Regression',color = 'b')
plt.scatter(X_13,train_labels,label = 'Train Data',color = 'g',alpha = 0.7 )
plt.title('LSTAT vs Price data and estimation')
plt.xlabel('Price')
plt.ylabel('LSTAT')


#find the MSE and show it
MSE = MSE(y_hat,train_labels)
print("MSE of the data is ",MSE)
print('The coefficients of the dataset are B0 =',float(B[0]),'  B1 =',float(B[1]))


   

#polynomail regression    
B_poly, y_hat_poly = linear_regression_polynomial(X_13,train_labels)
B_poly = B_poly.to_numpy()

#plot the figure for polynomial linear regression
plt.figure()
plt.scatter(X_13,train_labels,label = 'Train Data',color = 'g',alpha = 0.7 )
plt.scatter(X_13,y_hat_poly,label = 'Linear Regression',color = 'b',linewidths=1)
plt.title('LSTAT vs Price data and estimation polynomial regression')
plt.xlabel('Price')
plt.ylabel('LSTAT')

#calculate the MSE for polynomial regression
y_hat =y_hat_poly.to_numpy()
y = train_labels.to_numpy()
diff = y-y_hat
MSE_poly = np.mean(np.square(diff))
  
print("MSE of the data is ",MSE_poly)
print('The coefficients of the dataset with polynomial regressionare B0 =',B_poly[0],'  B1 =',B_poly[1],' B2 = ',B_poly[2])





#-------------------------   Q3    -----------------------------------------------

#take the data
data_train = pd.read_csv("question-3-features-train.csv")
data_test = pd.read_csv("question-3-features-test.csv")
train_labels= pd.read_csv("question-3-labels-train.csv")
test_labels= pd.read_csv("question-3-labels-test.csv")






learning_rate = 0.00001

data_train_normalized = normalize_data(data_train)
data_test_normalized = normalize_data(data_test)

# add bias to the train and test data and normalize
data_train_bias = np.concatenate([np.ones_like(train_labels),data_train_normalized],axis = 1)
data_test_bias = np.concatenate([np.ones_like(test_labels),data_test_normalized],axis = 1)




#FULL BATCH GRADIENT DESCENT



#initiali,ztaions
Y_real_test = test_labels.to_numpy()
Y_real_train = train_labels.to_numpy()
W = np.zeros(4).reshape(-1,1)


#update for 1000 batches

for epoch in range (1000):   
    V = data_train_bias @ W #find wX
    h_teta = sigm(V) # find the probability of a sample to be one
    delta_w = learning_rate* np.transpose(data_train_bias )@ (Y_real_train- h_teta.reshape(-1,1))/712
    
    W = W + delta_w


#find the test results
V_2 = data_test_bias @ W #find wX
prediction_test = np.ones_like(test_labels)
for i in range (len(test_labels)):
    if V[i] < 0:
        prediction_test[i] = 0
        
 #find acccuracy   
acc = 0        
for k in range(len(Y_real_test)):
    if prediction_test[k] == Y_real_test[k]:
        acc += 1
      
acc = acc/ len(Y_real_test)       
    
 #print the values  
score1= score_logistic(Y_real_test,prediction_test) 
print(score1)
    


#MINIBATCH GARDIENT DESCENT


Y_real_train = train_labels.to_numpy()
W = np.zeros(4).reshape(-1,1)



minibatch_size = 8

#update for 1000 batches
for l in range(1000):
    
    for epoch in range (int(712/minibatch_size)):
        
       
        
        batch_data = data_train_bias[epoch* minibatch_size: (epoch+1)*minibatch_size]
        batch_labels = Y_real_train[epoch* minibatch_size: (epoch+1)*minibatch_size]
        
        
        V = batch_data @ W #find wX
        h_teta = sigm(V) # find the probability of a sample to be one
        delta_w = learning_rate* np.transpose(batch_data )@ (batch_labels- h_teta.reshape(-1,1))/712
        
        W = W + delta_w


#find the test results
V_2 = data_test_bias @ W #find wX
prediction_train = np.ones_like(test_labels)


for i in range (len(V)):
    if V[i] < 0:
        prediction_train[i] = 0
        
 #find acccuracy   
acc = 0        
for k in range(len(Y_real_test)):
    if prediction_train[k] == Y_real_test[k]:
        acc += 1
      
acc = acc/ len(Y_real_test)       
    
 #print the values  
score1= score_logistic(Y_real_test,prediction_train) 
print(score1)



























