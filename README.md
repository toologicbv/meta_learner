## meta-learner 

#### Probabilistic meta learner that uses adaptive computation time to optimize loss functions


- **current version11:**  
	- pytorch version 0.2.0 (with CUDA 8.0);
	- can be used to optimize regression functions;
	- optimize regression functions by fitting Student-t distributions to the individual data points;
	- simple neural networks (e.g. 1-layer and 2-layer MLPs).


- Run `python train_optimizer.py --use_cuda --problem=mlp --functions_per_epoch=2048 --batch_size=128 --lr=1e-4 --learner=meta_act --hidden_size=40 --eval_freq=10 --max_epoch=150`

AI masterthesis. Iterative learning. Meta learning. Computational time adaptive learning to learn approach 



##### Synopsis
The master thesis project combines the work of Andrychowicz et al.  (Learning to learn by gradient descent by gradient descent, 2016) and Alex Graves (Adaptive Computation Time in RNNs, 2016). We develop a model that learns a task dependent optimization algorithm and approximates a posterior halting distribution over time steps by means of optimizing a variational lower bound on the log-likelihood. We hope to show that our ACT meta-learner uses on average less computational steps towards an optimum than the baseline LSTM optimizer from Andrychowicz et al. and that the posterior halting distribution can be exploited to dynamically determine the optimization step when computation should stop.


##### References
[Adaptive Computation Time for Recurrent Neural Networks](http://arxiv.org/abs/1603.08983) Graves, 2016

[Learning to learn by Gradient descent by gradient descent](https://arxiv.org/pdf/1606.04474.pdf) Andrychowicz et al, 2016
