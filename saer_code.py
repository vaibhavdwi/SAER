# Import Python packages for the program
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from google.colab import drive
drive.mount("/content/gdrive")

movies_data = pd.read_csv('/content/gdrive/My Drive/google collab/ml-1m/ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users_data = pd.read_csv('/content/gdrive/My Drive/google collab/ml-1m/ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings_data = pd.read_csv('/content/gdrive/My Drive/google collab/ml-1m/ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

movies_data.head()
users_data.head()



ratings_data.head()

training_dataset = pd.read_csv('/content/gdrive/My Drive/google collab/ml-100k/ml-100k/u1.base', delimiter = '\t')
training_dataset = np.array(training_dataset, dtype = 'int')
testing_dataset = pd.read_csv('/content/gdrive/My Drive/google collab/ml-100k/ml-100k/u1.test', delimiter = '\t')
testing_dataset = np.array(testing_dataset, dtype = 'int')

# Get the no. of users and movies from the data
no_users = int(max(max(training_dataset[:,0]), max(testing_dataset[:,0])))
no_movies = int(max(max(training_dataset[:,1]), max(testing_dataset[:,1])))
no_ratings = int(max(max(training_dataset[:,1]), max(testing_dataset[:,2])))

# Converting the dataset into array with all the users in lines and all movies in columns
def convert(dataset):
    new_dataset = []
    for uid_users in range(1, no_users + 1):
        uid_movies = dataset[:,1][dataset[:,0] == uid_users]
        uid_ratings = dataset[:,2][dataset[:,0] == uid_users]
        ratings_data = np.zeros(no_movies)
        ratings_data[uid_movies - 1] = uid_ratings
        new_dataset.append(list(ratings_data))
    return new_dataset
training_dataset = convert(training_dataset)
testing_dataset = convert(testing_dataset)

# Now we Convert the dataset into Torch tensors
training_dataset = torch.FloatTensor(training_dataset)
testing_dataset = torch.FloatTensor(testing_dataset)

# Now we Create the architecture of the Sparse Auto encoder
class SAER(nn.Module):
    def __init__(actfn, ):
        super(SAER, actfn).__init__()
        actfn.f1 = nn.Linear(no_movies, 20)
        actfn.f2 = nn.Linear(20, 10)
        actfn.f3 = nn.Linear(10, 20)
        actfn.f4 = nn.Linear(20, no_movies)
        actfn.activation = nn.Sigmoid()
    def forward(actfn, y):
        y = actfn.activation(actfn.f1(y))
        y = actfn.activation(actfn.f2(y))
        y = actfn.activation(actfn.f3(y))
        y = actfn.f4(y)
        return y
saer = SAER()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(saer.parameters(), lr = 0.01, weight_decay = 0.5)

no_of_epoch = 25
for epoch in range(1, no_of_epoch+1):
    training_loss = 0
    n = 0.
    for uid_user in range(no_users):
        input_var = Variable(training_dataset[uid_user]).unsqueeze(0)
        target_var = input_var.clone()
        ratings_target = target_var[:, :no_movies]
        if torch.sum(target_var.data > 0) > 0:
            output_var = saer(input_var)
            ratings_output = output_var[:, :no_movies]
            target_var.require_grad = False
            output_var[target_var == 0] = 0
            Errloss = criterion(ratings_output, ratings_target)
            avg_corrector = no_movies/float(torch.sum(target_var.data > 0) + 1e-10)
            Errloss.backward()
            training_loss += np.sqrt(Errloss.data*avg_corrector)
            n += 1.
            optimizer.step()
    print('epoch: '+str(epoch)+' Errloss: '+str(training_loss/n))

# Testing the SAER Model
testing_loss = 0
n = 0.
for uid_user in range(no_users):
    input_var = Variable(training_dataset[uid_user]).unsqueeze(0)
    target_var = Variable(testing_dataset[uid_user]).unsqueeze(0)
    ratings_target = target_var[:, :no_movies]
    if torch.sum(target_var.data > 0) > 0:
        output_var = saer(input_var)
        ratings_output = output_var[:, :no_movies]
        target_var.require_grad = False
        output_var[target_var == 0] = 0
        Errloss = criterion(ratings_output, ratings_target)
        avg_corrector = no_movies/float(torch.sum(target_var.data > 0) + 1e-10)
        testing_loss += np.sqrt(Errloss.data*avg_corrector)
        n += 1.
print('testing loss: '+str(testing_loss/n))

user_uid = 0
movie_title_name = movies_data.iloc[:no_movies, 1:2]
user_given_rating = training_dataset.data.numpy()[user_uid, :].reshape(-1,1)
user_target_rating = testing_dataset.data.numpy()[user_uid, :].reshape(-1,1)

user_target_rating[user_target_rating>0]

user_input_name = Variable(training_dataset[user_uid]).unsqueeze(0)
predicted_rating = saer(user_input_name)
predicted_rating = np.round(predicted_rating.data.numpy().reshape(-1,1), 2)

user_input_name = user_input_name.data.numpy().reshape(-1,1)
result_arr = np.hstack([movie_title_name, user_input_name, user_target_rating, predicted_rating])
result_data = pd.DataFrame(data=result_arr, columns=['Movie Name', 'User input', 'Target Rating', 'Predicted Rating'])

result_data.head()

result_data[result_data['Target Rating'] > 0]

#*End of Code#
