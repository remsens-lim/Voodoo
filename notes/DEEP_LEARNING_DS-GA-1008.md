DEEP LEARNING

**DS-GA 1008 · SPRING 2020 · [NYU CENTER FOR DATA SCIENCE](http://cds.nyu.edu/)**

| INSTRUCTORS                                               | Yann LeCun & Alfredo Canziani                                |
| --------------------------------------------------------- | ------------------------------------------------------------ |
| LECTURES                                                  | Mondays 16:55 – 18:35, [GCASL C95](http://library.nyu.edu/services/campus-media/classrooms/gcasl-c95/) |
| PRACTICA                                                  | Tuesdays 19:10 – 20:00, [GCASL C95](http://library.nyu.edu/services/campus-media/classrooms/gcasl-c95/) |
| [PIAZZA](https://piazza.com/nyu/spring2020/dsga1008/home) | Access code: `DLSP20`                                        |
| MATERIAL                                                  | [Google Drive](https://bitly.com/DLSP20), [Notebooks](https://github.com/Atcold/pytorch-Deep-Learning) |

## Description

This course concerns the latest techniques in deep learning and  representation learning, focusing on supervised and unsupervised deep  learning, embedding methods, metric learning, convolutional and  recurrent nets, with applications to computer vision, natural language  understanding, and speech recognition. The prerequisites include: [DS-GA 1001 Intro to Data Science](https://cds.nyu.edu/academics/ms-curriculum/) or a graduate-level machine learning course.



## Course Notes

UPDATES from 10MRZ2020:

- https://www.youtube.com/watch?v=0bMe_vCZo30
- the loss function is not convex, has local minima and sattle points which makes it difficult to optimise, not well understood
- roughly 2xn_categories=batch_size https://www.youtube.com/watch?v=d9vdh3b787Y at ~22min
- output layer softmax: normalises the output, make them all positive, make them look like probabilitie
-  now you want to maximise the probability that the model gives you the correct answer (log-softmax
- problems with sigmoid functions: https://youtu.be/d9vdh3b787Y?t=4574
- back propagation in practice (some infos, guidelines):https://youtu.be/d9vdh3b787Y?t=5006	
- choosing the right output layer activation is crucial ( especially in pytorch where (for categorisation) log_softmax is assumed!) otherwise always softmax ( test it for tensor flow) https://youtu.be/d9vdh3b787Y?t=5055 
-  mini batch ( in one minibatch you. Want roughly the same number of samples for every class you want to predict)
- check again on normalising/standartizing data, nicely explained here: https://youtu.be/d9vdh3b787Y?t=5225 
- could standardisation help getting rid of the gravity wave influence in Punta?
- proper weight initialisation is also very crucial (l1 regularisation not at the beginning of training)
- loss function MSE vs CE : https://youtu.be/WAn6lip5oWk?t=2167 
- neural network training clearly explained: https://youtu.be/WAn6lip5oWk?t=2250 
- comparison between ReLU and tanh: https://youtu.be/WAn6lip5oWk?t=3147 

 