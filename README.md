## meta-learner 

#### Probabilistic meta learner that uses adaptive computation time to optimize loss functions


- **current version11:**  
	- can be used to optimize regression functions;
	- optimize regression functions by fitting Student-t distributions to the individual data points;
	- simple neural networks (e.g. 1-layer and 2-layer MLPs).



AI masterthesis. Iterative learning. Meta learning. Computational time adaptive learning to learn approach 



##### Synopsis
The master thesis project combines the work of Andrychowicz et al.  (Learning to learn by gradient descent by gradient descent, 2016) and Alex Graves (Adaptive Computation Time in RNNs, 2016). We develop a model that learns a task dependent optimization algorithm and approximates a posterior halting distribution over time steps by means of optimizing a variational lower bound on the log-likelihood. We hope to show that our ACT meta-learner uses on average less computational steps towards an optimum than the baseline LSTM optimizer from Andrychowicz et al. and that the posterior halting distribution can be exploited to dynamically determine the optimization step when computation should stop.


##### References
[Adaptive Computation Time for Recurrent Neural Networks](http://arxiv.org/abs/1603.08983) Graves 2016
>>>>>>> 0c80f297044d1efad1c444558c1b027138b51ce9
