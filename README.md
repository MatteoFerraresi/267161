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

RFM analysis is typically used to identify the most valuable customers in a dataset, and it is often used in conjunction with other customer segmentation techniques. To perform RFM analysis, the data is typically divided into quantiles (e.g., top 20%, next 30%, etc.), and customers are assigned a score based on their ranking in each of the three categories. The scores are then combined to create a total RFM score, which can be used to identify the most valuable customers.

RFM analysis can be used to identify customers who are likely to respond to marketing campaigns, to target specific customer segments with personalized marketing messages, and to identify potential customers that may stop using or paying for a service.

It is generally a good idea to scale your data when using k-means clustering, as the algorithm relies on the distances between points in the data. If the features in your data have different units or scales, then this can affect the distances between points and potentially bias the clusters that are formed. By scaling the data, you can ensure that all of the features are on the same scale and the distances between points are more representative of the "true" distances between the points.

There are many ways to scale data, but a common method is to use standardization, which involves subtracting the mean of each feature from each data point and then dividing by the standard deviation. This ensures that the resulting scaled data has a mean of 0 and a standard deviation of 1.

It is important to note that you should only scale the features, not the target variable (if you have one). Additionally, it is usually recommended to scale the data before running k-means, as scaling the data after running the algorithm can change the clusters that are formed.

After plotting the RFM dataset, we can deduce that we have a positive correlation in between frequency and monetary value, due to the fact that the more a customer purchases the more he spends in total; so the more frequency value goes up, the more monetary value will go up too.

______

After performing an EDA and preprocessing the dataset, we used the following methods based on RFM that will help us explore data, plot it, analyze it, and have a more clear understanding of what's going on:
```bash
1. K-means
2. Hierarchical Clustering (HC)
3. Gaussian Mixture Model (GMM)
4. Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
```

Once we have identified the clusters, we describe the properties of each cluster in terms of the RFM scores and any other relevant variables and also describe the properties of the customers belonging to each cluster to better understand their characteristics and needs.

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
______

Succeeding, we imported the libraries and the dataset: logical and necessary step to provide a functional code. 

We then decided to conduct some data cleaning: startedwith the check of empty columns, then with the removal of duplicates (double checked to make sure everything is on track), changed the data type in the date columns so that it matches the supported data type, created a column "month_order" for data exploration, checked from when to when did the purchases go (so what time interval we are looking for), and finally split the data based on its datatype ("only_nymeric", "only_object", "only_time")

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


# **EXPERIMENTAL DESIGN**


# **RESULTS**


# **CONCLUSIONS**