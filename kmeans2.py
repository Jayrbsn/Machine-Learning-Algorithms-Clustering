#Resource: https://www.youtube.com/watch?v=EItlUEPCIzM
#Resource: https://towardsdatascience.com/k-means-clustering-for-beginners-ea2256154109
#Resource: https://stackoverflow.com/questions/42169892/is-there-a-way-to-limit-the-amount-of-while-loops-or-input-loops

#Some hints on how to start, as well as guidance on things which may trip you up, have been added to this file.
#You will have to add more code that just the hints provided here for the full implementation.
#You will also have to import relevant libraries for graph plotting and maths functions.
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.metrics import pairwise_distances_argmin
from collections import Counter, defaultdict

#Allow user to set amount of clusters and iterations

num_of_clusters = int(input("Enter desired number of clusters: "))
iterations = int(input("Enter desired number of iterations: "))

# Define a function that reads data in from the csv files  
# HINT: http://docs.python.org/2/library/csv.html. 
# HINT2: Remember that CSV files are comma separated, so you should use a "," as a delimiter. 
# HINT3: Ensure you are reading the csv file in the correct mode.

def read_csv():
    x = []
    y = []
    countries = []
    x_label = ""
    y_label = ""
    with open('dataBoth.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        lines = 0
        for row in reader:
            if lines >= 1:
                print(', '.join(row))
                x.append(float(row[1]))
                y.append(float(row[2]))
                countries.append(row[0])
                lines += 1
            else:
                x_label = row[1]
                y_label = row[2]
                print(', '.join(row))
                lines += 1
    return x, y, x_label, y_label, countries

x, y, x_label, y_label, countries = read_csv()

#Call np.vstack() method which is used to stack the sequence of input arrays vertically to make a single array
#In other words we combine x and y into a 2D list of pairs
#Plot and display scatter plot

X = np.vstack((x, y)).T

plt.xlabel(x_label)
plt.ylabel(y_label)
plt.scatter(x, y, color='green')
plt.show()

# Implement the k-means algorithm, using appropriate looping for the number of iterations
# --- find the closest centroid to each point and assign the point to that centroid's cluster
# --- calculate the new mean of all points in that cluster
# --- visualise (optional, but useful to see the changes)
#---- repeat

def find_clusters(X, n_clusters, rseed=2):
    #Create randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    count = 0
    
    #While loop to try and find convergence
    #Note count variable is set to 0 and increases by one with every iteration of this loop
    #The loop continues until count == iterations as chosen by user
    #Assign labels based on closest center
    #Use pairwise_distances_argmin method to  calculate the squared distance between each point and the 
    # centroid to which it belongs based on example from resource at top of sheet and as per compulsary task 2 
    # instructions 
    #Find new centers from means of points
    print("\nConverging centres:")
    while count < iterations:
        
        count += 1
        labels = pairwise_distances_argmin(X, centers)
        new_centers = np.array([X[labels == i].mean(0) for i in
        range(n_clusters)])

        if np.all(centers == new_centers):
            centers = new_centers
        
        #Print out this sum once on each iteration, and you can watch the objective function converge
        # as per compulsary task 2 instrucions
        print(centers)
        print()

    return centers, labels

#Revisualize scatter plot to see the changes

centers, labels = find_clusters(X, num_of_clusters)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.title('K-Means clustering of countries by birth rate vs life expectancy')
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.show()

# Print out the results for questions
#1.) The number of countries belonging to each cluster
#2.) The list of countries belonging to each cluster
#3.) The mean Life Expectancy and Birth Rate for each cluster

print("\nNumber of countries in each cluster:")
print(Counter(labels))

clusters_indices = defaultdict(list)
for index, c in enumerate(labels):
    clusters_indices[c].append(index)

x = 0
while x < num_of_clusters:
    print("\nCluster " + str(x + 1))
    print("----------")
    for i in clusters_indices[x]:
        print(countries[i])
    print("----------")
    print("Mean birth rate:")
    print(centers[x][0])
    print("Mean life expectancy:")
    print(centers[x][1])
    x+=1