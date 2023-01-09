# **MACHINE LEARNING PROJECT:** CUSTOMER SEGMENTATION

## Members
```bash
Matteo Ferraresi - 
Giulia Formiconi -  
Mohamed Ali Ben Belhassen - 273771
```

# **INTRODUCTION**

We are studying a large company's subsidiary resident in Brazil, and the goal is to identify the ideal partitions and assign each user in the dataset to one of them. The data we have concerns users, sellers, payments, among other things. 

First of all, we started by performing an exploratory data analysis (EDA). It is a process of analyzing and summarizing a dataset in order to understand its main characteristics and identify patterns or trends that may not be immediately obvious. EDA is an important step, as it helps to identify potential problems with the data and to formulate hypotheses about the relationships among the variables.

EDA typically involves visualizing the data using plots and charts, calculating descriptive statistics, and identifying patterns and trends in the data. It is an iterative process, as insights gained from one analysis may lead to further questions and further analysis.

It is a very important tool for gaining a deeper understanding of the data and for identifying patterns and trends that may not be immediately obvious. It is also an important step in the data science workflow, as it helps to identify potential problems with the data and to formulate hypotheses about the relationships among the variables.

______

Following, we started building our model using the RFM analysis for customer segmentation.

Customer segmentation is the process of dividing a customer base into smaller groups with similar characteristics. The goal of customer segmentation is to identify and understand the needs and characteristics of different groups of customers in order to tailor marketing, sales, and service efforts to each group.

Our objective is to segment customers in clusters depending on their behavior. 

What is RFM analysis? RFM stands for Recency, Frequency, and Monetary Value, and it is a customer segmentation technique used in marketing. Its analysis is based on the idea that the value of a customer to a business is determined by the recency of their last purchase (recency), the number of purchases they have made (frequency), and the total amount of money they have spent (monetary value).

RFM analysis is typically used to identify the most valuable customers in a dataset, and it is often used in conjunction with other clustering algorithms that are used for customer segmentation. 

RFM analysis can be used to identify customers who are likely to respond to marketing campaigns, to target specific customer segments with personalized marketing messages, and to identify potential customers that may stop using or paying for a service.

Despite scaling the data is a good idea, we decided not to do it because it compromised frequency values. Moreover, the silhouette score with scaled data didn't improve

After plotting the RFM dataset, we can deduce that we have a positive correlation in between frequency and monetary value, due to the fact that the more a customer purchases the more he spends in total; so the more frequency value goes up, the more monetary value will go up too.

______

After performing an EDA and preprocessing the dataset, we used the following methods based on RFM that will help us explore data, plot it, analyze it, and have a more clear understanding of what's going on:
```bash
1. K-means
2. Hierarchical Clustering (HC)
3. Gaussian Mixture Model (GMM)
4. Spectral Clustering (SC)
```

Once we have identified the clusters, we describe the properties of each cluster in terms of the used algorithm of the RFM.

Let's look into these algorithms deeper.

# **METHODS**

We chose to implement the following algortihms:

- **K-means:** clustering algorithm that divides a group of data points into clusters, where each cluster is represented by its centroid (a point that is the average of all the points in the cluster). The goal of k-means is to partition the data points into a predefined number of clusters in a way that minimizes the sum of the squared distances between each data point and the centroid of its cluster. To do this, the algorithm assigns each data point to the cluster whose centroid is closest to it, and then it updates the centroids by recomputing them as the mean of all the data points in the cluster.

- **Hierarchical Clustering:** method of cluster analysis that aims to build a hierarchy of clusters. It does this by creating a tree-like diagram, called a dendrogram, that shows the relationships among the data points and the clusters they belong to. There are two main types of hierarchical clustering: agglomerative and divisive. Agglomerative hierarchical clustering starts by treating each data point as a separate cluster and then iteratively combines the closest clusters until all the data points are in the same cluster, while divisive hierarchical clustering, on the other hand, starts by treating all the data points as a single cluster and then iteratively divides the cluster into smaller clusters until each data point is in a separate cluster. Hierarchical clustering is a good choice when you don't know how many clusters there are in the data or when you want to visualize the relationships among the clusters. 
In this case, we are going to opt for Agglomerative hierarchical clustering: one advantage of this type of hierarchical clustering is that it is easier to implement and is generally more efficient than divisive hierarchical clustering, especially for large datasets. It start with a partition of the data into individual clusters, and then iteratively merge the most similar clusters until a single cluster is obtained.

- **Gaussian Mixture Model:** GMM is a probabilistic model that assumes that the underlying data is generated from a mixture of a finite number of Gaussian distributions with unknown parameters. Each mixture component is a Gaussian distribution with a unique set of mean and covariance parameters, and each data point is associated with a weight that indicates the probability that it was generated by a particular mixture component. GMMs can be used for a variety of tasks, including density estimation, clustering, and classification.

- **SPECTRAL:** Spectral clustering is a method for clustering data points into groups, or clusters. The basic idea of spectral clustering is to construct a similarity graph of the data points, where each point is represented as a vertex in the graph and edges between vertices represent some measure of similarity between the points. The graph is then decomposed using techniques from linear algebra, such as singular value decomposition, to identify clusters of points that are more closely connected to each other. These clusters can then be used to assign the points to different groups.
Spectral clustering has a number of advantages over other clustering methods. It is highly scalable and can handle large datasets, and it can identify clusters that are not necessarily convex or uniformly sized. It is also relatively robust to noise and outliers in the data. However, it can be sensitive to the choice of similarity measure used to construct the graph, and it may not always produce the best results for all types of data.

Our graphs were plotted in 3D because we had 3 variables (RFM), and it was the best option to visualize clusters the best way possible.
As a evaluation metric we used silhouette score: it's a measure of how similar an object is to its own cluster compared to other clusters. It ranges from -1 to 1, with a high value indicating that the object is well matched to its own cluster and poorly matched to neighboring clusters. A score of 0 indicates that the object is on or very close to the decision boundary between two clusters.
The silhouette score is a good validation metric for clustering because it takes into account both the cohesion of the cluster, which is the average distance between the objects within the same cluster, and the separation between different clusters, which is the average distance between the objects of different clusters. A high silhouette score indicates that the clustering algorithm has done a good job of finding clusters that are both well-separated and internally cohesive. In general, this tool is useful for evaluating the performance of a clustering algorithm, especially when the true cluster assignments are not known. It can help you choose the number of clusters to use and compare the performance of different clustering algorithms.

We saw that the best silhouette score was the one for hierarchical clustering.

______

Succeeding, we imported the libraries and the dataset: logical and necessary step to provide a functional code. 

We then decided to conduct some data cleaning: started with the check of empty columns, then with the removal of duplicates (double checked to make sure everything is on track), changed the data type in the date columns so that it matches the supported data type, created a column "month_order" for data exploration, checked from when to when did the purchases go (so what time interval we are looking for), and finally split the data based on its datatype ("only_nymeric", "only_object", "only_time")

Next, we took care of the exploratory data analysis and created the following heatmap:

![heatmap](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/hitmap.png) 

We got some insights that could be useful to the company!

![orderstatus](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/order_status.png)
______
![paymenttype](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/payment_type.png)
______
![customerstatedistribution](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/cust_state_distr.png)
______
![sellerstatedistribution](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/seller_state_distr.png)
______
![orderitemiddistribution](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/order_item_id_distr.png)
______
![top20bestsellingproducts](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/top_20_best_selling_products.png)
______
![top20mostpopularcities](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/top_20_most_popular_cities.png)
______
![top10customerbasedonorderamount](https://github.com/MatteoFerraresi/267161/blob/main/images/top_10_cust_based_on_order_amount.png)
______
![top10customerbasedonspending](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/top_10_cust_based_on_spending.png)
______
![top10fastestproductcategoryordertime](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/top_10_fsatest_product_category_order_time.png)
______
![top10slowestproductcategoryordertime](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/top_10_slowest_product_category_order_time.png)

When we looked at these two plots, it was immediately clear that they made sense. This is because products with fast shipping are typically smaller and lighter (such as baby clothes), while items like office furniture take much longer to ship.
______
![monthlyorder](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/monthly_order.png)
______
![monthlyrevenue](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/monthly_revenue.png)
______
![monthlyactiveusers](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/monthly_active_users.png)

Here we see basic details about the subsidiary's monthly activity. It is evident that the month with the most orders also had the highest revenue and highest number of active users.

# **EXPERIMENTAL DESIGN**

Let's plot the densities:

![recency](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/recency.png)
![frequency](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/frequency.png)
![monetary](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/monetary.png)

There is a relationship between customers who make purchases more often and customers who spend the most overall. This is likely due to the fact that the more purchases a customer makes, the more they are spending in total.

![recencyfrequency](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/monefreq.png)

This graph demonstrates that the company has both a growing customer base and an increasing amount of revenue. The increase in revenue can be seen by the higher amount of money being spent in recent days, and the increase in the number of customers can be inferred from the increased frequency of orders (as shown in the previously mentioned graphs).

For customers who make more than one purchase, the average time between purchases is 2.5 months. On average, customers tend to spend around 250 dollars and only make one purchase. It is important to note that this data may be slightly skewed due to the presence of outliers - there are a few customers who have spent significantly more than the average amount or made significantly more purchases than the average number. For example, there is one customer who spent nearly 30,000 dollars and another who made 13 purchases.

To evaluate the quality of the clusters we obtained, we used  silhouette score: it measures the separation between clusters by considering the distance between the samples within each cluster and the distance between the samples and the closest cluster. We used this score to determine which combination of hyperparameters resulted in the best separation of the clusters. 

______

K-Means CLUSTERING

The elbow method was utilized to identify the optimal number of clusters (k) for the k-means clustering algorithm. This method helps to determine the most suitable value for k by examining the changes in error within the clusters as k increases.

![elbowmethod](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/elbowmethod.png)

To help visualize the data, we used a technique called the elbow method with the yellowbrick library.

![distortionscoreelbow](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/distortionscoreelbow.png)

![rfm](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/rfm.png)

# **RESULTS**

It is time to calculate the silhouette score. 

### KMEANS:

We found that for kmeans, the silhouette score was equal to approximatively 0.82. As a matter of fact, we applied different scaling techniques (MinMax scaling and Standard scaling) and also used unscaled data to see which configuration would result in the highest silhouette score for our data. When using Standard scaling, we obtained a silhouette score of approximately 0.49. With MinMax scaling, the score was slightly higher at 0.5. However, the highest silhouette score was achieved with the unscaled data, at 0.82. As a result, we decided to use the unscaled data to construct our clusters.

The silhouette score for k-means is: 0.821611

![kmeans](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/kmeansclusters.png)

![kmeans3d](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/kmeans3d.png)

### HIERARCHICAL CLUSTERING:

Hierarchical clustering is another method for grouping customers into clusters. This approach works by creating a tree-like structure called a dendrogram, in which the root node represents the entire dataset and branches are formed to create smaller clusters. The dendrogram can be used to determine the appropriate number of clusters to analyze. With hierarchical clustering, a subset of similar data is organized into a tree structure, with the root node representing the entire dataset. Branches are created from the root node to form smaller clusters, and the number of clusters can be determined by analyzing the dendrogram.

The silhouette score for HC is: 0.871694

![dendrogram](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/dendrogram.png)

![agglomerative](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/hcclusters.png)

![agglomerative3d](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/newplottttt.png)

### GAUSSIAN MIXTURE MODEL:

One advantage of GMM is that it is more flexible than K-Means when it comes to cluster covariance. The standard deviation parameter allows clusters to have any elliptical shape, rather than just circular shapes. GMM also allows for multiple clusters (mixed membership) for a single data point because it uses probabilities.

However, GMM tends to be slower than K-Means because it requires more iterations to converge.
To determine the optimal number of clusters for our data, we plotted the silhouette score while varying the number of clusters (k) from 1 to 10. This allowed us to see which value of k resulted in the highest silhouette value, indicating the best separation of the clusters.

We see 2, 3 and 4 have very high silhouette scores, hence we choose 4 clusters (the most reasonable choice).

If we had used the k-means algorithm to initialize the Gaussian Mixture Model (GMM), we would have obtained a silhouette score of 0.82. However, by using random initialization instead, we obtained a lower silhouette score but more diverse and potentially more informative results. These results will be discussed in more detail later on.

In this case, using the k-means algorithm to initialize the GMM would have resulted in a silhouette score that is similar to the score obtained with the k-means algorithm alone. In addition, the resulting clusters would have been largely the same as those obtained with k-means. Therefore, we decided to use the random initialization method instead in order to obtain more varied and potentially more informative results.

![gmm](https://github.com/MatteoFerraresi/267161/blob/main/images/gmmclusters.png)

The silhouette score for GMM is: 0.464953

![gmm2](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/gmmclusters2.png)

![gmm3d](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/newplot.png)

### SPECTRAL CLUSTERING:

One advantage of spectral clustering is that it does not assume that the data is distributed in a particular way or that the clusters have a specific shape. This allows it to perform well with a wide range of data shapes. Spectral clustering also does not require the actual data, but only the similarity and distance matrix or the Laplacian matrix. This makes it possible to cluster one-dimensional data.

However, spectral clustering does have some limitations. One disadvantage is that the number of clusters must be specified in advance, although heuristics can be used to determine an appropriate value. In addition, spectral clustering can be computationally expensive, although there are frameworks and algorithms available to mitigate this issue.

By calculating the silhouette score, we found that the latter is equal to 0.310082

![spectral](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/spectralclusters.png)

![spectral3d](https://github.com/MatteoFerraresi/267161/blob/main/images/spectral3d.png)


______

We found ourselves with the following values:
![valuessilhouette](https://raw.githubusercontent.com/MatteoFerraresi/267161/main/images/conclusions.jpg)

The best silhouette's scores are the one for Hierarchical Clustering and K-means, while the worst ones are for GMM and Spectral Clustering.

# **CONCLUSIONS**

There are some tips that the Brazilian subsidiary can implement:

- The four customer segments are: at-risk spenders, low spenders, mid spenders, and loyal spenders. 
- Basically, the subsidiary should focus the email campaign on low and at-risk customers, in order to convince them to spend more and more frequently. 
- For the mid ones, the company should gives more options about products which costs less. 
- For the loyal customers, the firm should promote the usual products which are complex ones but also the more simple ones.

______

THANK YOU VERY MUCH!