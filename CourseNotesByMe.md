### Dropout

This technique depends on zeroing out the output of the hidden neurons by probability of any fraction between(0.1 ~ 0.9) and by doing so this will reduces complex co-adaptations of neurons since a neuron can not rely on the presence of particular other neurons. It is, therefore, forced to learn more robust features that are useful in conjunction with many different random subsets of the other neurons.

Dropout roughly doubles the number of iterations required to converge.


### Filters

- Using small filters (3 × 3) (which is the smallest size to capture the notion of left/right, up/down, center).
- Using (1 × 1) convolution filters, which can be seen as a linear transformation of the input channels (followed by non-linearity), aslo it increase the nonlinearity of the decision function without affecting the receptive fields of the conv layers.
- Why to use say a stack of 3 (3 × 3) filters instead of 1 (7 × 7) filter?
	- First, we incorporate three non-linear rectification layers instead of a single one, which makes the decision function more discriminative.
	- Second, we decrease the number of parameters


###  Local Response Normalisation (LRN)

implements the lateral inhibition. This layer is useful when we are dealing with ReLU neurons. Why is that? Because ReLU neurons have unbounded activations and we need LRN to normalize that. We want to detect high frequency features with a large response. If we normalize around the local neighborhood of the excited neuron, it becomes even more sensitive as compared to its neighbors. At the same time, it will dampen the responses that are uniformly large in any given local neighborhood. If all the values are large, then normalizing those values will diminish all of them. So basically we want to encourage some kind of inhibition and boost the neurons with relatively larger activations. 

[source](https://prateekvjoshi.com/2016/04/05/what-is-local-response-normalization-in-convolutional-neural-networks/)


### Do Deeper Networks means higher accuracy?

When deeper networks are able to start converging, a degradation problem has been exposed: with the network depth increasing, accuracy gets saturated (which might be unsurprising) and then degrades rapidly. Unexpectedly, such degradation is not caused by overfitting, and adding more layers to a suitably deep model leads to higher training error


### why (1 × 1) Convolution?

1. Network in a network as it learns complex non-linear relations about a slice of a volume
2. Reducing number of channels i.e. (28 × 28 × 192) * (1 × 1 × 92) {#Filters=32} = (28 × 28 × 32)



