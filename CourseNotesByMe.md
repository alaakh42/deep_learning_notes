### Dropout

This technique depends on zeroing out the output of the hidden neurons by probability of any fraction between(0.1 ~ 0.9) and by doing so this will reduces complex co-adaptations of neurons since a neuron can not rely on the presence of particular other neurons. It is, therefore, forced to learn more robust features that are useful in conjunction with many different random subsets of the other neurons.

Dropout roughly doubles the number of iterations required to converge.


### Filters

- Using small filters `(3 × 3)` (which is the smallest size to capture the notion of left/right, up/down, center).
- Using `(1 × 1)` convolution filters, which can be seen as a linear transformation of the input channels (followed by non-linearity), aslo it increase the nonlinearity of the decision function without affecting the receptive fields of the conv layers.
- Why to use say a stack of 3 `(3 × 3)` filters instead of 1 `(7 × 7)` filter?
	- First, we incorporate three non-linear rectification layers instead of a single one, which makes the decision function more discriminative.
	- Second, we decrease the number of parameters


###  Local Response Normalisation (LRN)

implements the lateral inhibition. This layer is useful when we are dealing with ReLU neurons. Why is that? Because ReLU neurons have unbounded activations and we need LRN to normalize that. We want to detect high frequency features with a large response. If we normalize around the local neighborhood of the excited neuron, it becomes even more sensitive as compared to its neighbors. At the same time, it will dampen the responses that are uniformly large in any given local neighborhood. If all the values are large, then normalizing those values will diminish all of them. So basically we want to encourage some kind of inhibition and boost the neurons with relatively larger activations. 

[source](https://prateekvjoshi.com/2016/04/05/what-is-local-response-normalization-in-convolutional-neural-networks/)


### Do Deeper Networks means higher accuracy?

When deeper networks are able to start converging, a degradation problem has been exposed: with the network depth increasing, accuracy gets saturated (which might be unsurprising) and then degrades rapidly. Unexpectedly, such degradation is not caused by overfitting, and adding more layers to a suitably deep model leads to higher training error.


### why (1 × 1) Convolution?

1. Network in a network as it learns complex non-linear relations about a slice of a volume
2. Reducing number of channels i.e. `(28 × 28 × 192)` * `(1 × 1 × 92)` `#Filters=32` = `(28 × 28 × 32)`


### IOU and Non-Max Supression

- __IOU__ aka. (Intersection over Union) in which we calculate the intersection area between every different window and the object then divide it by the union of the total windows and the object, it is used to tell how good a window in detecting the object when more than one window claims the same object. the higher IOU the better

- __Non-max Supression Algorithm__ if two bounding boxes have the same IOU then you will pick the one with the highest Pc (probability of object exsistence)

- __Anchor Boxes__ let you encode multible objects in the same square(overlapping objects), 


### One Shot Learning

In whih you need to learn a model that can recognize say an image given only one sample. This can be done using:
1. __Similarity Function__ where you calculate the degree of differnce between two images which is done using 
a `Siamese Network` where you input the two images then by applying back propagation to the `Triplet loss` function that calculates a number. if number is small then images are the same if it is large they are different


### Why RNN?

1. Because input and output can have different lengthd in different examples
2. we don't share features learned across different positions aka. keeps track of previous parts of a sequence. so `w`, `b` are shared between all time steps


### Why GRU?

Because of the vanishing/ exploiding gradients problem where a simple RNN cannot keep track of long sequences. so Gated Recurrent Units acts a `memory cell` to keep track of all pevious information in a sequence while in the same time using a `gate` to decide whether to store a token in a sequence or not


### Why LSTM?

It is another variation of GRU but the difference is that LSTM has a `forget` gate


### Word Embedding

- __Negative sampling__ helps in shortening the learning time of word embedding, it depends on picking one positive pair of 2 highly correlated word vectors while picking a few random negative samples (in other words the negative samples are ones that don't exist in the specified window size) so you only train each context word for #times = #negative_examples + 1 postive_example
	- __Word2vec__: Skipgrams and Continous BoW
	- __GLove__   : an extension of Word2vec where also the number of word occuranc is taking into account
	- __Fastext__ : an extension of Word2vec model , treats each word as a sequence composed of char ngrams so the word vector is made up of the sum of this char ngrams. 
		- It has better word vecs for rare words because even if words are rare their characters are diffenitely shared with other words
		- doesn't suffer OOV (out if vocabulary) error because we can always construct a word vector for a new word by summing its char ngrams vectors
		- slower to generate so you have to choose well max/min char ngram length


### Transfer Learning

Transfer Learning in NLP is useful for NER, Text Summurization but no so useful for Machine Translation and Language Modeling


### Hierachial Softmax



### BLEU Score

The bilengual evaluation understudy, is a metric for evaluating a generated sentence to a reference sentence. A perfect match results in a score of 1.0, whereas a perfect mismatch results in a score of 0.0. [source](https://machinelearningmastery.com/calculate-bleu-score-for-text-python/)




