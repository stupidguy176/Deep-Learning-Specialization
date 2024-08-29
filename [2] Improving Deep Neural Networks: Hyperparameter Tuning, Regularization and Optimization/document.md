
## Main assigments


### Module 2 Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization



### Documents


### Github
The experience of predecessors


### Heroes


### Ideas

https://www.coursera.org/learn/deep-neural-network/lecture/bqUgf/yoshua-bengio-interview




### Notes
https://www.coursera.org/learn/deep-neural-network?specialization=deep-learning

https://community.deeplearning.ai/t/dls-course-2-lecture-notes/11866

học cái văn phong

học dưới áp lực

# Cách học

Nhìn sơ qua khóa học

- lấy tổng quan kiến thức sẽ học
- học được gì từ khóa học này?
- nó tiếp nói j với khóa học trước?
- dự án thực tế hay bài tập cần làm

Làm bài kiểm tra trước

- biết mình yếu gì, cần học gì

Nhảy cmn zo dự án thực tế

# Keywords

bias and variance - bias và phương sai

trade-off? - đánh đổi

phương sai và độ lệch chuẩn

Frobenius norn - Chuẩn Frobenius 

Regularizing your neural network

- **weight decay** (phân rã trọng số)
- dropout
- 

# Week 1

- Documents
    
    https://quizlet.com/vn/812606815/course-2-flash-cards/
    
    ![screencapture-coursera-org-learn-deep-neural-network-exam-yTz20-practical-aspects-of-deep-learning-attempt-2024-08-24-17_41_09.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7dca1731-068d-4997-814f-8d55bce2cdd6/5398cd06-5f15-4e25-8065-d477d61bd87b/screencapture-coursera-org-learn-deep-neural-network-exam-yTz20-practical-aspects-of-deep-learning-attempt-2024-08-24-17_41_09.png)
    

Basic recipe for machine learning

How does regularization prevent overfitting?
Regularizing your neural network

- Dropout regularization
- Implementing dropout (“Inverted dropout”)

Why does drop-out work?
-  Intuition: Can’t rely on any one feature, so have to spread out weights.

Other regularization methods

Setting up your optimization problem

- Normalizing inputs
- Vanishing/exploding gradients
- Numerical approximation of gradients
    - Checking your derivative computation
- Gradient Checking
- Gradient Checking implementation notes

# Week 2

Optimization Algorithms

Mini-batch gradient descent

Batch vs. mini-batch gradient descent

Understanding mini-batch gradient descent

- Training with mini batch gradient descent
- Choosing your mini-batch size

Exponentially weighted averages

Understanding exponentially weighted averages

- Exponentially weighted averages
- Implementing exponentially weighted averages

Bias correction in exponentially weighted average

Gradient descent with momentum

- Implementation details

RMSprop

Adam optimization algorithm

- Hyperparameters choice:

Learning rate decay

- learning rate decay methods

The problem of local optima

- Local optima in neural networks
- Problem of plateaus

# Week 3

Hyperparameter tuning

- Tuning process
    - Hyperparameters
    - Try random values: Don’t use a grid
    - Coarse to fine
- Using an appropriate	scale to pick hyperparameters
    - Picking hyperparameters at random
    - Appropriate scale for hyperparameters
    - Hyperparameters for exponentially weighted averages
- Hyperparameters tuning in practice: Pandas vs. Caviar
    - Re-test hyperparameters occasionally
- Batch Normalization
    - Normalizing	activations	
    in	a	network
        - Normalizing inputs to speed up learning
        - Implementing Batch Norm
    - Fitting Batch Norm into a	neural network
        - Adding Batch Norm to a network
        - Working with mini-batches
        - Implementing gradient descent
    - Why does Batch	Norm work?
        - Learning on shifting input distribution
        - Why this is a problem with neural networks?
        - Batch Norm as regularization
- Multi-class classification
    - Softmax	regression
        - Softmax layer
        - Softmax examples
- Programming Frameworks
    - Deep Learning frameworks
    - TensorFlow















