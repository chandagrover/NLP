
# <font color = 'purple'>Distributional  Semantics

## <font color='brown'>1) Semantics: </font>

Semantics is the linguistic and philosophical study of meaning, in language, programming languages and formal logics.It is the study of meaning of words/symbols used in a sentence. i.e What is the relation between symbols and what do they denote.
 

__e.g__ _Jon told mary that the train moved out of the station at 3 O'clock_.

This sentence is in natural language containing multiple words/token. In semantics, we want to know what does each word represent sepearately. Each of the word is some symbol in one sense and when they combine together in a sentence, what meaning do they contribute to sentence. This is one of the very vast field in NLP where research is going on. In general we want how computer can produce such semantic representation and so how machine can interpret some sort of semantic representation, that's why the idea of __computational semantics__ comes.

## <font color='brown'> 2) Computational Semantics </font>

The study of how to automate the process of constructing and reasoning with meaning representations of natural language expressions. Methods in compuatational semantics generally fall in two categories:


1) __Formal Semantics__ : Construction of precise mathematical models of the relations between expressions in a natural language and the world i.e How do I construct various mathematical models that can tell me what is the relation between various 
expressions in the language and also relate them to whatever there is in there in the word e.g _Predicate Logic_

2) __Distributional Semantics__ : The study of statistical patterns of human word usage to extract semantics. i.e Can I study the statistical patterns of human word usage to extract semantics. Can I see how humans are using different words in the their language to find out what is their semantics and this is what we will explain. What is the type of distributional semantics, how they capture that and how can i use that for certain meaningful applications. We will cover distributional distributional semantics here. This field of distributional semantics is mainly built upon this hypothesis of distributional hypothesis. Few Famous quotes of distributional hypothesis:

![semantics.PNG](attachment:semantics.PNG)

###  <font color='brown'>2.1) Distributional hypothesis :Basic Intuition

a) _"The meaning of a word is its usage in language."_ (Wittgenstein, 1953) i.e We can know the meaning of a word if we see how it is being used in the language.

b) _"You know a word by the company it keeps"._ (Firth, 1957) Company of a word means what are the other words it occurs with in our corpus or in our language and that tells us about the word.

Both quotes tells in one sense that word meaning (whatever it might be) is reflected in linguistic distributions i.e the way the word has been used in the language, that will tell us the meaning of a word. 

###  <font color='brown'> 2.2) Distributional Semantics :Linguistic Perspective

c) _"Words that occur in the same contexts tend to have similar meanings."_ (Zelling Harris, 1968)
It says, two words have similar or different meanings if we can somehow measure and capture their context and if the context in which they occur are very very similar or different. So, if they are semantically similar, they tend to have similar distribution patterns.


d) _"If linguistic is to deal with meaning, it can only do so through distributional analysis."_ (Zelling Harris)

e) _"If we consider words or morphemes A and B to be more different in meaning than A and C, then we will often find that the distributions of A and B are more different than the distributions of A and C. In other words, difference in meaning correlates with the difference of distribution."_ (Zelling Harris)

![p1.PNG](attachment:p1.PNG)

We will use  Zelling Harris perspectives to represent the distribution patterns of words and use that to find out similarity across words. The semantics that we will capture will be __diffferential not referential__. We cannot represent what is the meaning of car with this sort of representation. Rather, we are capturing how similar or different two meanings are. Therefore, this is the differential sort of understanding of semantics and not the referential in distributional semantics.

###  <font color='brown'>2.3) Distributional Semantics : Cognitive Perspective

__Contextual representation:__ A word's contextual representation is an abstract congnitive structure that accumulates from encounters with the word in various linguistic contexts.i.e We learn new words based on contextual cues

__e.g__ He filled the __wampimuk__ with the substance, passed it around and we all drunk some_.

We did not know the meaning of wampimuk earlier, but as we saw its usage in more and more contexts, we can built certain sort of understanding that it might be soe sort of a container or some sort of a glass which can be used for filling up the substances.

__Another example:__ _We found a little __wampimuk__ sleeping behind the tree._ So, with this example we can say, wampimuk might be some small animal. 

<font color = 'pink'>__Therefore from the above all different perspectives , we consider semantics as the distribution patterns of words in language.__</font>


## <font color='brown'> 3) Distributional Sematic Models (DSM)

The models that captures distribution pattern of words  in language and find the semantics from there are called distributional semantic models.

To build this, we are given some sort of corpus data on how different words are used in the language and from there we build some semantic model distribution. In general, more the data given, better is the distribution representation. 

The semantic content is represented by a vector. Vectors are obtained through the statistical analysis of the linguistic contexts of a word.


### <font color='brown'>3.1) Distribution Semantics: General Intuition

Distributions are vectors denoting the words in a multidimensional semantic space, i.e. objects with a magnitude and a direction. The semantic space has dimensions which correspond to possible contexts, as gathered from a given corpus. We will represent every word in this semantics space. This is the distribution semantics. 

![vector%20space.PNG](attachment:vector%20space.PNG)

Here dimensions are context words i.e. eat and drive. We are representing cat, dog and car in the semantic space of eat and drive. Projection of a word on particular dimension denote how often a word occur in that particular dimension. We can clearly see that cat and dog occur more in eat dimension as compared to drive dimension whereas car appears in more contexts of drive than eat. Hence we can say cat and dog are much similar than cat and car. 

We can represent a word in as many dimensions we want to represent. e.g cat = [... dog 0.8, eat 0.7, joke 0.01, mansion 0.2.....]

###  <font color='brown'>3.2) Constructing Word Space:


1. Pick the words you are interested in : __target words__
2. Define a  __context window__, number of words surrounding target word. Context can be defiend in terms of documents, paragraphs or sentences.
3. Count number of times the target word co-occurs with the coontext words: <b> co-occurence matrix</b>.
4. Build vectors out of a (a function of) these co-occurence count.


![dataset.PNG](attachment:dataset.PNG)

![vectors.PNG](attachment:vectors.PNG)

Here we have 4 words as target words, 7 words as context words and matrix as (4*7) dimension. Each element in this matrix denote how often this target word occured with this context word within my context window. 

Let us take only two dimension of semantic space, say, goal and transport.Also let's represent 4 target words in the semantic space of two dimension as follows:



![2dim.PNG](attachment:2dim.PNG)

We can calculate similarity between two vectors using dot product:e.g. Automobile and Car tend to be more similar than automobile and Soccer or automobile and football. Similarity between all word pairs is represented below: 

![similar.PNG](attachment:similar.PNG)

###  <font color='brown'>3.3) Vector Space model without distributional similarity:

__One hot Encoding:__  Here we consider each word as a vector of dimension as the size of vocabulary and only the component of vector is set to one corresponding to the word. Rest all components of word are zero.

__e.g.__ Motel = [0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
Here we can see only one component of vector is one corresponding to the word. Rest all are zero.

__Problem:__ We can’t capture similarity or differences between two words with one hot encoding because most of the dimensions are zero except one for each word. Therefore when we take dot product between these two, we will get zero as the product.


![OHE.PNG](attachment:OHE.PNG)

###  <font color='brown'>3.4) Vector Space Model with Distributional similarity:

We know a word by the company it keeps. i.e. Instead of putting one corresponding to one dimension, only,  we try to put various distribution across all dimensions.

#### <font color='brown'> Building a DSM step-by-step

_(Linguistic steps)_
1. Pre-process a corpus (to define targets and contexts)
2. Select the targets and the contexts.

_(Mathematical Steps:)_
3.	Count the target-context co-occurrences
4.	Weight the contexts (optional)
5.	Build the distributional matrix
6.	Reduce the matrix dimension (optional)
7.	Compute the vector distances on the (reduced) matrix.


####  <font color='brown'>Parameter Space: 

There are a number of parameters that need to be fixed building DSM: 

a.	Which type of context we will use.<br>
b.	Which weighting scheme we will use.<br>
c.	Which similarity measure shall we use.<br>

A specific parameter setting determines a particular type of DSM.


## <font color='brown'> 4) Application of Distribution Semantics: 

__Query Expansion:__ Addressing Term Mismatch: Let us consider a use rquery: <I> Insurance cover which pays for long term care.</I> A relevant document may contain terms different from the actual user query. Some relevant words concerning this query : {medicare, premiums,  insurers}

__Using DSM for Query Expansion:__ Given a user query, reformulate it using related terms to enhance the retrieval performance. The distributional vectors for the query terms are computed. Expanded query is obtained by a linear combination or a functional combination of these vectors.


## <font color = 'brown'> 5) Similarity Measures 

![simmeasure.PNG](attachment:simmeasure.PNG)


### <font color ='brown'>5.1) Binary Vectors

5.1a)	__Dice Coefficient:__ Let X and Y denote the binary distribution vectors for words X’ and Y’.

![dice.PNG](attachment:dice.PNG)

![Phone2.jpg](attachment:Phone2.jpg)

__5.1b) Jaccard Coefficient:__


Let X and Y denote the binary distribution vector for words X’ and Y’.

![p3.PNG](attachment:p3.PNG)

<font color = 'green'> <b>Jaccard and Dice are different in the sense that if intersection between two words is less , then jaccard penalizes more as compared to dice. </b> </font>

__5.1c) Overlap Coefficient:__


 It is used when we want to know if one word is completely subsumed into another, then we use overlap coffiecient.

![Phone4.jpg](attachment:Phone4.jpg)

<font color ='green'><B> Jaccard coefficient penalizes for small number of shared entries, while overlap coefficient uses the concept of inclusion.  </B></font>

### <font color ='brown'> 5.2) Vector Spaces

Let X and Y denote the distributional vectors for words X’ and Y’ having n-dimensional real values.
X= [x1,x2,…..,xn]
Y=[y1,y2,……, yn]

![cosineeuc.PNG](attachment:cosineeuc.PNG)

The difference between Cosine and Euclidean distance is that, if the vectors are not normalized, then both cosine similarity and Euclidean distance will give different values, but if the vectors are normalized, then they will give same similarity measures.

### <font color ='brown'> 5.3) Probability Disribution

Let p and q denote the probability distributions corresponding to two distributional vectors. 

![probdist.PNG](attachment:probdist.PNG)

##  <font color ='brown'> 6) Relational Similarity

 - Two pairs (a,b) and (c,d) are said to be relationally similar if they have many similar relations. 
          Example- king:man and queen:woman
 - Realtional semilarities can be exploited using vector arithmetic which is show in following example. Here we are performing vector addition of vectors of 'king' and 'woman' and subtacting 'queen' vector and the result we are getting is most similar to 'men'
 - Read more on relational semantics and word analogy on https://levyomer.wordpress.com/2014/04/25/linguistic-regularities-in-sparse-and-explicit-word-representations/ [4]
 - Dataset description: Dataset taken is pre trained vectors trained on part of Google News dataset (about 100 billion words). The model contains 300-dimensional vectors for 3 million words and phrases. The phrases were obtained using a simple data-driven approach described in. We have extracted about 50-70 vectors out of original model because it is easy for explanation purpose.


```python
#Example showing vector arithmetic to exploit word similarities
#Precomputed vectors data taken from: https://code.google.com/archive/p/word2vec/
from scipy import spatial
import numpy as np
dictionary = dict()
with open("extracted_vectors.txt","r") as f: #sentences is the data set we have used
    for line in f:
#         print(line)
        temp = line.strip("\n").split(",")
#         print(temp)
        vector = temp[1].split(";")
#         print(vector)
        vector = [float(i) for i in vector]
        dictionary[temp[0]] = vector
#         print(vector)
dictionary
king = np.asarray(dictionary['king'])
queen = np.asarray(dictionary['queen'])
man = np.asarray(dictionary['man'])
woman = np.asarray(dictionary['woman'])
water = np.asarray(dictionary['water'])

# simiarity between (vector(king)-vector(man)) and (vector(queen)-vector(woman))
similarity = 1 - spatial.distance.cosine(king-man,queen-woman)
print ('Similarity: ')
print (similarity)
```

    Similarity: 
    0.7580350281015321
    


```python
#Graph plot using matplotlib
from scipy import spatial
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('Similarity based on cosine distance')
names = [['king','queen'],['man','woman'],['king','water']]
k=1
plt.plot([0],[0])
for i in names:
    y = [k,k]
    x = [0.5,0.5]
    dist = spatial.distance.cosine(dictionary[i[0]],dictionary[i[1]])
    dist = dist/2
    x[0] = x[0] - dist
    x[1] = x[1] + dist
    myvec = np.array([x,y])
    j=0
    for xy in zip(x, y):    
        ax.annotate(str(i[j]), xy=xy, textcoords='data',va='bottom')
        j = j + 1
    plt.plot(myvec[0,],myvec[1,])
    j = 0
    k = k+1
plt.plot([0],[4])    
plt.show()
```


![png](output_70_0.png)


##  <font color ='brown'> 7)  References:


[1] Distributional semantics https://en.wikipedia.org/wiki/Distributional_semantics <br>
[2] Pawan Goyal, Lecture slides, Speech and natural language processing (CS60057), Autumn 2016, IIT Kharagpur.<br>
[3] https://github.com/krishnamrith12/NotebooksNLP/blob/master/8.Introduction_to_WordNet_and_Word_semantics.ipynb

