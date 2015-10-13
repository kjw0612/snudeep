## cifar training settings

- cifar-lenet: original lenet trainig
- cifar-lenet-1: original lenet training

- cifar-maxout-init-1: maxout training with max-pooling initialization, but overfitting (stopped before finishing)
- cifar-maxout-init-2: maxout with max-pool initialization, with lr = [0.01*ones(1,30) 0.005*ones(1,10) 0.001*ones(1,5)], wd = 0.0001

-cifar-maxout-[1~7]: maxout with gaussian random initialization
