# Fake News Detection Using Graph Neural Networks

## Project Overview

This project explores the use of Graph Neural Networks (GNNs) for fake news detection. By leveraging the graph structure of news articles and their relationships, the model captures relational information for more accurate classification.

---

## Features and Workflow

### 1. Dataset Preparation
- **Data Types**:
  - Textual data: Tokenized and transformed into embeddings (e.g., TF-IDF or pre-trained embeddings).
  - Graph data: Represents relationships between news articles using adjacency matrices.
- **Preprocessing**:
  - Feature extraction for nodes (articles).
  - Adjacency matrices to encode relationships.

### 2. Graph Construction
- **Nodes**: Represent individual news articles, enriched with textual features.
- **Edges**: Define relationships (e.g., citations, similarities) between articles.
- **Graph Representation**: Graph objects created using libraries like `PyTorch Geometric`.

### 3. GNN Model
- **Architecture**:
  - Graph Convolutional Layers to aggregate node information from neighbors.
  - Fully connected layers for classification.
- **Output**:
  - Predicts the likelihood of an article being fake or real.
- **Training**:
  - Optimizer: Adam.
  - Loss function: Binary cross-entropy.

### 4. Evaluation
- **Metrics**:
  - Accuracy, Precision, Recall, F1-Score, ROC-AUC.
- **Visualization**:
  - Loss curves, ROC curves, confusion matrix, and metric trends.

---

## Visualizations and Results

### **Graph Representations**

#### **Training Graph**


- **Description**: Nodes represent articles; edges represent relationships.
- **Observation**: Dense connectivity highlights interdependencies.

#### **Testing Graph**
![Testing Graph](<path-to-testing-graph-image>)
- **Description**: Nodes represent unseen articles.
- **Observation**: Sparse structure to evaluate the model's generalization.

---

### **Training and Test Loss**
![image](https://github.com/user-attachments/assets/77a7bfc6-b521-4ee3-8849-d49a460ef7de)
- **Description**: Tracks training and test loss over epochs.
- **Observation**:
  - Training loss consistently decreases.
  - Test loss fluctuates, indicating potential overfitting in certain epochs.

---

### **Metrics Over Epochs**

![image](https://github.com/user-attachments/assets/b67457a1-c848-4016-99bb-714d0fcf770c)

- **Description**: Tracks Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
- **Observation**:
  - Fluctuations in metrics indicate the model's sensitivity to data variability.
  - Overall improvement in key metrics over training epochs.

---

### **Confusion Matrix**
![image](https://github.com/user-attachments/assets/f38e9251-d3b1-47a8-aae2-66c17297b8a6)

- **Description**: Summarizes the model's prediction performance.
- **Observation**:
  - High true positives (1636) and true negatives (1830).
  - Low false positives (80) and false negatives (280).

---

### **ROC Curve**
![image](https://github.com/user-attachments/assets/5709e02d-4d1e-4103-af3e-4d36b1cfb717)

- **Description**: Shows the trade-off between sensitivity and specificity.
- **Observation**:
  - High ROC-AUC score demonstrates strong classification ability.

---

### **Sample Predictions**
| Predicted Logit | Predicted Label | True Label |
|------------------|-----------------|------------|
| 0.699477         | 1.0             | 1          |
| 0.507549         | 1.0             | 1          |
| 0.270993         | 0.0             | 0          |
| 0.526134         | 1.0             | 1          |
| 0.092196         | 0.0             | 0          |
| 0.261437         | 0.0             | 0          |
| 0.205597         | 0.0             | 0          |
| 0.805393         | 1.0             | 1          |
| 0.095358         | 0.0             | 0          |
| 0.196156         | 0.0             | 0          |

---

## Discussion and Insights

### **Performance**:
- High ROC-AUC and accurate predictions suggest effective utilization of relational data.
- Precision-recall trade-offs show the model prioritizes reducing false negatives.

### **Challenges**:
- Fluctuations in test loss and metrics highlight generalization challenges.
- Sparse graph structures in testing require the model to rely on indirect relationships.

### **Future Work**:
- Add richer features and additional relationship types.
- Explore advanced GNN architectures (e.g., Graph Attention Networks).
- Implement techniques to handle overfitting and improve metric stability.

---

## How to Use

### **Clone the Repository**
```
git clone <repository-url>
cd fake-news-detection-gnn
```
