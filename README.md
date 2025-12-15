# Deep-learning-Assignment
Deep Learning assignment implemented in Google Colab .

This repository contains multiple implementations of Deep Learning and Reinforcement Learning models, completed as part of an academic assignment. Each section focuses on a different learning paradigm, model architecture, or dataset, along with improvements made over baseline implementations.

📌 Technologies Used

Python
NumPy
Pandas
Matplotlib
TensorFlow / Keras
Scikit-learn
Google Colab

📑 Code Execution Order

1. Tic Tac Toe
2. Deep Reinforcement Learning
3. RNN
4. CatDog
5. AlexNet
6. LSTM

🔹 1. Tic Tac Toe (Reinforcement Learning)
Description: This implementation trains an AI agent to play Tic Tac Toe using Reinforcement Learning. The agent learns optimal strategies through self-play using an ε-greedy policy and later allows human vs AI gameplay.

Modifications Made:
a) Fixed initialization bug in player symbol

b) Added epsilon decay for better exploration–exploitation balance

c) Improved reward propagation logic

d) Added win-rate tracking during training

e) Visualized learning performance using graphs

f) Refactored code for clarity and stability

Outcome: After training, the agent demonstrates effective learning behavior and competitive gameplay against a human player.

🔹 2. Deep Reinforcement Learning (Q-Learning)
Description: This section implements Q-learning, where an agent learns optimal actions by interacting with an environment and updating a Q-table based on rewards.

Modifications Made:
a) Refactored Q-learning logic for clarity

b) Added epsilon-greedy policy for balanced exploration

c) Removed redundant function definitions

d) Improved reward propagation logic

e) Added convergence graph to visualize learning

f) Simplified environment–agent interaction

g) Smoothed reward curve to clearly show learning convergence

Outcome: The enhanced implementation clearly demonstrates convergence behavior and optimal policy learning.

🔹 3. Recurrent Neural Network (RNN)
Description: This section implements a Recurrent Neural Network (RNN) to model sequential data and capture temporal dependencies.

Modifications Made:
a) Added train–validation split to evaluate generalization

b) Introduced Dropout layers to prevent overfitting

c) Changed optimizer from Adam to RMSprop for better sequence learning

d) Increased RNN depth to improve accuracy

e) Added loss and accuracy plots for visualization

f) Improved code readability and modularity

Outcome: The updated RNN model shows improved stability, better generalization, and clearer training insights.

🔹 4. CatDog (Image Classification)
Description: This section performs image classification using a CNN  Originally implemented as binary classification (Cat vs Dog), the model was later extended to multi-class image classification by replacing the dataset with the Flowers dataset.

Modifications Made:
a) Replaced Cat–Dog dataset with multi-class Flowers dataset

b) Implemented data augmentation to improve generalization

c) Added Dropout layer to reduce overfitting

d) Changed optimizer to Adam with a lower learning rate

e) Converted binary classification to categorical classification

f) Added training and validation accuracy/loss graphs

g) Displayed predicted and actual labels with images for better understanding

Outcome: The model achieves better generalization and provides improved visualization for interpretation of predictions.

🔹 5. AlexNet
Description: This section focuses on improving an RNN-based text/sequence generation model by correcting architectural and sampling-related issues that affected output quality.

Modifications Made:
a) Replaced ReLU activation with tanh, which is standard for RNNs

b) Fixed the number of RNN units instead of using text length

c) Increased the number of neurons to strengthen the model

d) Replaced argmax-based sampling with temperature sampling

e) Improved overall model stability and output diversity

Outcome: The modified model generates more natural, less repetitive outputs and is more suitable for demonstrations and viva explanations.

🔹 6. Long Short-Term Memory (LSTM)
Description: This project implements time series forecasting using an LSTM neural network to predict airline passenger counts based on historical data.

Modifications Made:
a) Implemented proper time-series preprocessing

b) Used LSTM architecture for long-term dependency learning

c) Applied Adam optimizer with MSE loss

d) Evaluated performance using RMSE

e) Visualized training and testing predictions

Outcome: The LSTM model successfully captures temporal trends and produces accurate forecasts for airline passenger data.
