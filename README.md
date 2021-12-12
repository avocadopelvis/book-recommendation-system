# Book Recommendation System

 A book recommendation system based on popularity, correlation and collaborative filtering.
 
 # Project Overview
 
Recommendation systems are information filtering systems that deal with the problem of information overload by filtering vital information fragments out of large amounts of dynamically generated information according to a user’s preferences, interest, or observed behavior about a certain item. A recommendation system has the ability to predict whether a particular user would prefer an item or not based on the user’s profile. <br/>

The aim of this project is to build a book recommendation system that can provide interesting book recommendations to the user based on a book's popularity or the user's interests/preferences. <br/>

The eventual result of the book recommendation system is not to make accurate predictions but instead to provide difficult to quantify insightful book recommendations.

# Data Description

This project uses a dataset containing 1.1 million ratings of 270,000 books by 90,000 users with the ratings being on a scale from 1 to 10. The dataset was obtained from Book-Crossings (http://www2.informatik.uni-freiburg.de/~cziegler/BX/) and it comprises three tables for users, books and rating.

# Methodology

To build this recommendation system, I have taken three approaches:

## 1. Popularity Based Recommendation System.

With its simplicity, this is the most basic recommendation system which offers generalized recommendation to every user based on their popularity. In a bookstore, if a certain book is popular among its customers & is also critically acclaimed, in the scenario that a new customer walks in & asks for the best, they would be suggested to try that book too. The same is true for movies, shows, music, etc. Whatever is more popular among the general public, is more likely to be recommended to new customers too. <br/>

This type of recommendation system makes generalized recommendation not personalized, meaning that this system will not take into account the personal
preferences or choices, rather it would tell that this particular thing is liked by most of the users.

To build one, the count of user ratings were taken for different books & the top 10 rated books are displayed below:
![1](https://user-images.githubusercontent.com/92647313/145728235-e10f0923-a9b9-4fb7-a6f8-d27b7d0980d7.png)

The recommended books are:
![16](https://user-images.githubusercontent.com/92647313/145728827-eaadfc6f-be46-404e-90bc-44cf3a7d185f.png)

## 2. Correlation Based Recommendation System.

Correlation coefficients are used to measure how strong a relationship is between two variables. There are several types of correlation coefficient, but the most popular is Pearson’s. A Pearson correlation is a number between -1 and +1 that indicates to which extent 2 variables are linearly related. So in this case, it is the rating for two books. <br/>

First, the average rating & the number of ratings each book received were found and then sorted based on the rating count in descending order.
![2](https://user-images.githubusercontent.com/92647313/145728284-e7c5cdf1-4c98-4458-ba5d-34ce9b8c0024.png)

**Observations**: The book with the most rating counts isn't necessarily a highly rated book. As seen from the table above, the book with the most rating counts of '2502' only had a rating of '1.019584'. As a result, if recommendations were made solely based on rating counts, it is evident that mistakes would be made.

The ‘ratings’ table is then converted into a 2D matrix. The matrix is sparse since not every user rated every book.
![3](https://user-images.githubusercontent.com/92647313/145728325-686042cc-6419-49cf-82d6-7bc1dbbf9349.png)

To test the system, the book **'Divine Secrets of the Ya-Ya Sisterhood'** by Rebecca Wells was chosen which tells the story of the downward spiraling mother-daughter relationship of Vivian Walker and Siddalee Walker.<br/>

The following are the books correlated with the above mentioned book:
![4](https://user-images.githubusercontent.com/92647313/145728379-33d4b2bc-b977-48c3-9567-ce1a1684f649.png)

The recommended books are:
![5](https://user-images.githubusercontent.com/92647313/145728901-32eab0b6-b27f-44e2-9070-fb8a9d41a58b.png)

**White Oleander** by Janet Fitch tells the unforgettable story of Ingrid, a brilliant poet imprisoned for murder, and her daughter, Astrid, whose odyssey through a series of Los Angeles foster homes--each its own universe, with its own laws, its own dangers, its own hard lessons to be learned--becomes a redeeming and surprising journey of self-discovery. <br/>

**The Nanny Diaries** by Emma McLaughlin is a humorous but revealing novel about the experiences of a college girl, Nan, as she nannies for a wealthy family on the upper east side of Manhattan. <br/>

**The Secret Life of Bees** tells the story of Lily Owens, whose life has been shaped around the blurred memory of the afternoon her mother was killed. When Lily's fierce-hearted black "stand-in mother," Rosaleen, insults three of the deepest racists in town, Lily decides to spring them both free. They escape to Tiburon, South Carolina--a town that holds the secret to her mother's past. Taken in by an eccentric trio of black beekeeping sisters, Lily is introduced to their mesmerizing world of bees and honey, and the Black Madonna.

All the above mentioned books touch on the themes of motherhood & maternal love describing the different complexities of the relationship between a mother and daughter. The books also go into the intricacies of a character coming of age & focus on the psychological and moral growth or transition of a protagonist from youth to adulthood. Based on the above points, we can imply that our correlation based book recommendation system is working

## 3. Collaborative Filtering Based Recommendation System.
In Collaborative Filtering, we tend to find similar users and recommend what similar users like. In this type of recommendation system, we don’t use the features of the item to recommend it, rather we classify the users into the clusters of similar types, and recommend each user according to the preference of its cluster.

### a.)  Using K-Nearest Neighbors:
K-Nearest Neighbors (KNN) algorithm is a type of supervised ML algorithm which can be used for both classification as well as regression predictive problems. KNN algorithm assumes the similarity between the new case/data and available cases and puts the new case into the category that is most similar to the available categories. <br/>

Using this algorithm, clusters of similar users based on common book ratings can be found and predictions can be made using the average rating of the top-k nearest neighbors. <br/>

To look for popular books, the ‘books’ data was combined with the ‘ratings’ data. 
![6](https://user-images.githubusercontent.com/92647313/145728534-a3af23b6-6d96-46ea-8412-c1269caff677.png)

They were then grouped by the book titles & a new column for the total rating count was created.
![7](https://user-images.githubusercontent.com/92647313/145728557-87a85630-380a-4a98-9512-ec77da4103dd.png)

The ‘ratings’ data is combined with the total rating count data to find out which books are popular & to filter out the lesser-known books.
![8](https://user-images.githubusercontent.com/92647313/145728576-f6aaf026-e7e6-427b-9326-15018631a210.png)

Since there are many books in the dataset, they have been limited by setting the popularity threshold as '100' which ensures that books with 100 or more ratings will be selected.
![9](https://user-images.githubusercontent.com/92647313/145728628-1223c60e-c9c1-4892-89c3-130c16227813.png)

**Note**: To cope with the computing power of my machine, the users have been limited to those in the US & Canada. The user data is then combined with the rating data & the total rating count data. <br/>

The table is converted into a 2D matrix & the missing values are filled with zeroes since the distances between rating vectors will be calculated. The values of the matrix dataframe are then transformed into a scipy sparse matrix for more efficiency calculations <br/>

The algorithm used to compute the nearest neighbors is **'brute'** & the metric is **'cosine'** so that the algorithm will calculate the cosine similarity between rating vectors.
![10](https://user-images.githubusercontent.com/92647313/145729046-ce8a47ba-8c05-4b43-995c-47e8a208cb28.png)

### b.) Using Matrix Factorization:

Matrix factorization is a class of collaborative filtering algorithms used in recommender systems. Matrix factorization algorithms work by decomposing the user-item interaction matrix into the product of two lower dimensionality rectangular matrice. <br/>

**Singular value decomposition** (SVD) is used here. <br/>
The US-Canada users' rating table is converted into a utility matrix & the missing values are filled with zeros
![12](https://user-images.githubusercontent.com/92647313/145729826-369612c2-2b5c-4220-8b89-9ded81721b8e.png)

This utility matrix is transposed in order for the 'booktitle' & 'userIDs' to become rows &
columns respectively. After using TruncatedSVD to decompose it, it is fitted into the model for further dimensionality reduction. <br/>

Pearson's R correlation coefficient is calculated for every book pair in the final matrix.
![13](https://user-images.githubusercontent.com/92647313/145728768-dec4781b-65ba-42fd-b5f6-d35fc8f45aa7.png)

The book **'1984'** was picked in order to compare the results from the KNN algorithm
![14](https://user-images.githubusercontent.com/92647313/145728794-ae1c4000-d14b-47e4-b69f-cfcb8b0ccafa.png)

**1984** by George Orwell is a dystopian novel which tells of a terrifying vision of a totalitarian future in which everything and everyone is a slave to a tyrannical regime. <br/>

The recommended books using KNN are:
![11](https://user-images.githubusercontent.com/92647313/145729062-69e4c14d-00e6-4e46-b66b-774bc5042a03.png)

Animal Farm by George Orwell tells the story of a farm taken over by its overworked, mistreated animals. With flaming idealism and stirring slogans, they set out to create a paradise of progress, justice, and equality. Thus the stage is set for one of the mosttelling satiric fables ever penned –a razor-edged fairy tale for grown-ups that records the evolution from revolution against tyranny to a totalitarianism just as terrible. <br/>

**Brave New World** by Aldous Huxley is a dystopian novel largely set in a futuristic World State, inhabited by genetically modified citizens and an intelligence-based social hierarchy, the novel anticipates huge scientific advancements in reproductive technology, sleep-learning, psychological manipulation and classical conditioning that are combined to make a dystopian society which is challenged by only a single individual: the story's protagonist. <br/>

All of the above mentioned books deal with the subject of totalitarianism, propaganda & the idea of a dystopia. While Animal Farm is an allegory for the Russian Revolution of 1917, both 1984 & Brave New World tell of a future society in which governments have complete dictatorial control over people, while state control and conformity replace the freedoms of modern life and a person's right to the pursuit of happiness. I have personally read all three books in the past & so I can also attest to its similarities. <br/>

The recommended books using matrix factorization are:
![15](https://user-images.githubusercontent.com/92647313/145729387-5802c1ab-b1e1-4793-b507-6c0f8e485285.png)

As seen from the above highlighted texts, the recommendation system using matrix factorization also recommended the same books as the system using knn algorithm.

# Conclusion
From the recommendations made, it is apparent that the system was able to recommend books to a user that has yet to read/rate based on the popularity of a book, the similarity of a previously read/rated book or their respective ratings. With these recommendations, a successful book recommendation system has been created that can make predictions & recommend books to its users. Lastly, based on the system's ability to make recommendations, it can be concluded that the system can be effective to help retail book stores/sites improve their marketing strategies & control inventory which can potentially translate into higher sales & profits.

