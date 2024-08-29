
### Main points
- 
-

## Main assigments
- 
- 

## Module 1 Neural Networks and Deep Learning

Logistic Regression with a Neural Network Mindset

Planar Data Classification with One Hidden Layer

Building your Deep Neural Network: Step by Step

Deep Neural Network - Application


### Heroes

1 - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/dcm5r/geoffrey-hinton-interview

2 - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/eqiZZ/pieter-abbeel-interview

3 - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/WSia1/ian-goodfellow-interview



[Basic knowledge - Functions, Derivatives, Integrals ](https://www.notion.so/Basic-knowledge-Functions-Derivatives-Integrals-94dc20b0c6764d7aa8f82831eed9ccf1?pvs=21)

- Module 2
    
    https://community.deeplearning.ai/t/dls-course-1-lecture-notes/11862
    
    ## Logistic Regression as a Neural Network
    
    Logistic Regression Cost Function là gì?
    
    để đo lường sự khác biệt giữa các giá trị dự đoán của mô hình và các giá trị thực tế của dữ liệu huấn luyện. 
    
    ![Screenshot 2024-06-30 134936.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7dca1731-068d-4997-814f-8d55bce2cdd6/2af33861-fcf5-4aad-acaf-00d6984405e2/Screenshot_2024-06-30_134936.png)
    
    ![Screenshot 2024-06-30 140129.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7dca1731-068d-4997-814f-8d55bce2cdd6/f5b304cd-c895-4f99-a0cd-3a9e7626bb99/Screenshot_2024-06-30_140129.png)
    
    ![Screenshot 2024-06-30 140424.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7dca1731-068d-4997-814f-8d55bce2cdd6/66bd86d6-04fe-4f48-bb16-76789b05ab86/Screenshot_2024-06-30_140424.png)
    
    ![Screenshot_1.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7dca1731-068d-4997-814f-8d55bce2cdd6/6f600e72-2ef8-4c88-bc54-e2b035ab7cd7/Screenshot_1.png)
    
    ![Screenshot_2.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7dca1731-068d-4997-814f-8d55bce2cdd6/7f57b2da-7ab8-4403-a1fa-4be315cb7a24/Screenshot_2.png)
    
    ![Screenshot_3.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7dca1731-068d-4997-814f-8d55bce2cdd6/5675d884-e72b-486b-a0a7-3b5024805ec6/Screenshot_3.png)
    
    ![Screenshot_4.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7dca1731-068d-4997-814f-8d55bce2cdd6/5c0390e2-4172-4c02-91a9-9fedb2aa3c77/Screenshot_4.png)
    
    ---
    
    ### **Derivation of DL/dz (Optional)**
    
    https://community.deeplearning.ai/t/derivation-of-dl-dz/165
    
    ![Screenshot_1.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7dca1731-068d-4997-814f-8d55bce2cdd6/5daf2dcf-deb8-47bf-b37d-870cd7c60581/Screenshot_1.png)
    
    ---
    
    ## Python and Vectorization
    
    ![Screenshot_2.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7dca1731-068d-4997-814f-8d55bce2cdd6/01a3fccb-c1a2-4e14-bfb5-d722c8a6943d/Screenshot_2.png)
    
    ![Screenshot_2.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7dca1731-068d-4997-814f-8d55bce2cdd6/efde9e00-f990-4b9a-86b2-d479b083a574/Screenshot_2.png)
    
    ![Screenshot_1.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7dca1731-068d-4997-814f-8d55bce2cdd6/b217b94f-bc9f-4fc6-87dc-0762d4571598/Screenshot_1.png)
    
- Module 3
    
    
    https://community.deeplearning.ai/t/dls-course-1-lecture-notes/11862
    
    https://chatgpt.com/c/13646773-825f-4f28-b9ff-17e1109456ba
    
    https://miro.com/welcomeonboard/WG5iNHJrWFdaTG9aN2RWclEwSXQzb3FrZGlEV1FwRVIwamxZM2JLY3FnVTl4d1h2dVBoYXdqdkFGNDRHdE9rY3wzNDU4NzY0NTYxOTI3Njg2OTYwfDI=?share_link_id=719566656564
    
    https://www.coursera.org/learn/neural-networks-deep-learning/lecture/WSia1/ian-goodfellow-interview#
    
    https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html
    
    các ký hiệu tính toán?
    
    các node trong mạng được training như nào?
    
    bên trong 1 node có gì? nó tính toán j ở trong za?
    
    nhiều exmaple thì liên kết và training với nhau kiểu gì?
    
    hàm active làm j zai chời? nó dùng làm j?
    
    cost function, lost function, back and forward,
    
    gradient decent (forward propagation, back propagation…)
    
    **Reminder**: The general methodology to build a Neural Network is to:
    
    ```
    1. Define the neural network structure ( # of input units,  # of hidden units, etc).
    2. Initialize the model's parameters
    3. Loop:
        - Implement forward propagation
        - Compute loss
        - Implement backward propagation to get the gradients
        - Update parameters (gradient descent)
    ```
    
    - **4 - Neural Network mode**
        
        **4.1 - Defining the neural network structure**
        
        **4.2 - Initialize the model's parameters**
        
        **4.3 - The Loop**
        
        **4.4 - Compute the Cost**
        
        **4.5 - Implement Backpropagation**
        
        **4.6 - Update Parameters**
        
        **4.7 - Integration**
        
    - **5 - Test the Model**
        
        **5.1 - Predict**
        
        **5.2 - Test the Model on the Planar Dataset**
        
    
    **6 - Tuning hidden layer size (optional/ungraded exercise)**
    
    **7 - Performance on other datasets**
    
    ```
    logprobs = np.multiply(np.log(A2),m)
    cost = - np.dot(logprobs,Y)
    
    dZ2 = A2 - Y
    dW2 = (dW2*A1)/m
    db2 = np.sum(dZ2,axis=1,keepdism=True)/m
    
    dZ1 = (W2*dZ2)*(1 - np.power(A1, 2))
    dW1 = (dZ1*X)/m
    db1 = np.sum(dZ1,axis=1,keepdism=True)/m
    ```
    
- Module 4
    
    https://community.deeplearning.ai/t/feedforward-neural-networks-in-depth/98811
    
    https://quizlet.com/vn/786939791/dpl301-deep-learning-flash-cards/
    
    https://quizlet.com/vn/804674975/dpl-course1-week4-flash-cards/
    
    https://marcossilva.github.io/en/2019/06/24/coursera-deep-learning-notes-module-1-week-4.html
    
    doc tai lieu truoc file:///C:/Users/ADMIN/Downloads/C1_W4.pdf
    
    thu lam bai kiem tra truoc
    
    quay lai xem  noi dung bai hoc
    
    Deep L-layer Neural network
    
    What is a deep neural network?
    
    Deep neural network notation
    
    Forward propagation in a deep network
    
    Getting your matrix dimensions right
    
    Parameters W["l”] and b[”l"]
    
    Vectorized implementation
    
    Why deep representations?
    
    Intuition about deep representation
    Circuit theory and deep learning
    
    Building blocks of deep neural networks
    
    Forward and backward functions
    
    Forward and backward propagation
    
    Parameters vs Hyperparameters
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7dca1731-068d-4997-814f-8d55bce2cdd6/f416dab1-e1d3-4c36-8301-e3f6bb6aae63/image.png)
    
    - **Building your Deep Neural Network: Step by Step**
        
        ## **Table of Contents**
        
        - [1 - Packages](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A1/Building_your_Deep_Neural_Network_Step_by_Step.ipynb#1)
        - [2 - Outline](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A1/Building_your_Deep_Neural_Network_Step_by_Step.ipynb#2)
        - [3 - Initialization](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A1/Building_your_Deep_Neural_Network_Step_by_Step.ipynb#3)
            - [3.1 - 2-layer Neural Network](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A1/Building_your_Deep_Neural_Network_Step_by_Step.ipynb#3-1)
                - [Exercise 1 - initialize_parameters](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A1/Building_your_Deep_Neural_Network_Step_by_Step.ipynb#ex-1)
            - [3.2 - L-layer Neural Network](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A1/Building_your_Deep_Neural_Network_Step_by_Step.ipynb#3-2)
                - [Exercise 2 - initialize_parameters_deep](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A1/Building_your_Deep_Neural_Network_Step_by_Step.ipynb#ex-2)
        - [4 - Forward Propagation Module](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A1/Building_your_Deep_Neural_Network_Step_by_Step.ipynb#4)
            - [4.1 - Linear Forward](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A1/Building_your_Deep_Neural_Network_Step_by_Step.ipynb#4-1)
                - [Exercise 3 - linear_forward](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A1/Building_your_Deep_Neural_Network_Step_by_Step.ipynb#ex-3)
            - [4.2 - Linear-Activation Forward](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A1/Building_your_Deep_Neural_Network_Step_by_Step.ipynb#4-2)
                - [Exercise 4 - linear_activation_forward](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A1/Building_your_Deep_Neural_Network_Step_by_Step.ipynb#ex-4)
            - [4.3 - L-Layer Model](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A1/Building_your_Deep_Neural_Network_Step_by_Step.ipynb#4-3)
                - [Exercise 5 - L_model_forward](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A1/Building_your_Deep_Neural_Network_Step_by_Step.ipynb#ex-5)
        - [5 - Cost Function](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A1/Building_your_Deep_Neural_Network_Step_by_Step.ipynb#5)
            - [Exercise 6 - compute_cost](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A1/Building_your_Deep_Neural_Network_Step_by_Step.ipynb#ex-6)
        - [6 - Backward Propagation Module](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A1/Building_your_Deep_Neural_Network_Step_by_Step.ipynb#6)
            - [6.1 - Linear Backward](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A1/Building_your_Deep_Neural_Network_Step_by_Step.ipynb#6-1)
                - [Exercise 7 - linear_backward](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A1/Building_your_Deep_Neural_Network_Step_by_Step.ipynb#ex-7)
            - [6.2 - Linear-Activation Backward](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A1/Building_your_Deep_Neural_Network_Step_by_Step.ipynb#6-2)
                - [Exercise 8 - linear_activation_backward](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A1/Building_your_Deep_Neural_Network_Step_by_Step.ipynb#ex-8)
            - [6.3 - L-Model Backward](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A1/Building_your_Deep_Neural_Network_Step_by_Step.ipynb#6-3)
                - [Exercise 9 - L_model_backward](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A1/Building_your_Deep_Neural_Network_Step_by_Step.ipynb#ex-9)
            - [6.4 - Update Parameters](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A1/Building_your_Deep_Neural_Network_Step_by_Step.ipynb#6-4)
                - [Exercise 10 - update_parameters](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A1/Building_your_Deep_Neural_Network_Step_by_Step.ipynb#ex-10)
    - **Deep Neural Network for Image Classification: Application[¶](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A2/Deep%20Neural%20Network%20-%20Application.ipynb#Deep-Neural-Network-for-Image-Classification:-Application)**
        
        ## **Table of Contents**
        
        - [1 - Packages](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A2/Deep%20Neural%20Network%20-%20Application.ipynb#1)
        - [2 - Load and Process the Dataset](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A2/Deep%20Neural%20Network%20-%20Application.ipynb#2)
        - [3 - Model Architecture](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A2/Deep%20Neural%20Network%20-%20Application.ipynb#3)
            - [3.1 - 2-layer Neural Network](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A2/Deep%20Neural%20Network%20-%20Application.ipynb#3-1)
            - [3.2 - L-layer Deep Neural Network](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A2/Deep%20Neural%20Network%20-%20Application.ipynb#3-2)
            - [3.3 - General Methodology](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A2/Deep%20Neural%20Network%20-%20Application.ipynb#3-3)
        - [4 - Two-layer Neural Network](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A2/Deep%20Neural%20Network%20-%20Application.ipynb#4)
            - [Exercise 1 - two_layer_model](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A2/Deep%20Neural%20Network%20-%20Application.ipynb#ex-1)
            - [4.1 - Train the model](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A2/Deep%20Neural%20Network%20-%20Application.ipynb#4-1)
        - [5 - L-layer Neural Network](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A2/Deep%20Neural%20Network%20-%20Application.ipynb#5)
            - [Exercise 2 - L_layer_model](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A2/Deep%20Neural%20Network%20-%20Application.ipynb#ex-2)
            - [5.1 - Train the model](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A2/Deep%20Neural%20Network%20-%20Application.ipynb#5-1)
        - [6 - Results Analysis](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A2/Deep%20Neural%20Network%20-%20Application.ipynb#6)
        - [7 - Test with your own image (optional/ungraded exercise)](https://cywdabfkzgnf.labs.coursera.org/notebooks/release/W4A2/Deep%20Neural%20Network%20-%20Application.ipynb#7)

















