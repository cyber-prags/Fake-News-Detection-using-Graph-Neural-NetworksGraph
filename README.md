# **Fake News Detection Using Graph Neural Networks (GNNs)**

## **Project Overview**
This project explores the use of **Graph Neural Networks (GNNs)** for **fake news detection**. By leveraging the graph structure of news articles and their relationships, the model captures **relational information** for more accurate classification.

🚀 **Key Highlights**:
- Uses **Graph Attention Networks (GATConv)** to capture article relationships.
- Constructs graphs where **nodes represent news articles** and **edges define relationships**.
- Trained using **binary cross-entropy loss** and optimized via **Adam optimizer**.
- Evaluated using **accuracy, precision, recall, F1-score, and ROC-AUC**.

---

## **1️⃣ Features and Workflow**

### **1. Dataset Preparation**
- **Data Types**:
  - **Textual Data**: Tokenized and transformed into embeddings (TF-IDF, pre-trained embeddings).
  - **Graph Data**: Represents relationships between news articles using **adjacency matrices**.
- **Preprocessing Steps**:
  - Extracts **text-based features** for each news article.
  - Constructs **adjacency matrices** to encode relationships between articles.

---

### **2. Graph Construction**
- **Nodes**: Represent individual news articles, enriched with textual features.
- **Edges**: Define relationships based on:
  - **Citations** between articles.
  - **Textual similarities** between articles.
  - **Social media interactions** (e.g., Twitter-based dataset).
- **Graph Representation**:
  - Uses **NetworkX** and **PyTorch Geometric** for graph processing.
  - Converts raw **news articles into a structured graph**.

---

### **3. Graph Neural Network (GNN) Model**
- **Architecture**:
  - **Graph Attention Network (GATConv)** layers for message passing.
  - **Fully connected layers** for final classification.
- **Training**:
  - **Optimizer**: Adam (`lr=0.01`, `weight_decay=0.01`).
  - **Loss Function**: Binary Cross-Entropy Loss (`BCELoss`).
- **Prediction Output**:
  - Probability that an article is **fake (1) or real (0)**.

---

### **4. Model Evaluation**
- **Evaluation Metrics**:
  - **Accuracy**: Measures overall correctness.
  - **Precision & Recall**: Captures trade-offs in misclassifications.
  - **F1-Score**: Harmonic mean of precision and recall.
  - **ROC-AUC**: Assesses the model’s ability to distinguish between fake and real news.
- **Visualization**:
  - **Loss curves** over epochs.
  - **ROC curve** to show trade-off between sensitivity and specificity.
  - **Confusion matrix** to analyze model predictions.

---

## **2️⃣ Visualizations and Results**

### **Graph Representations**
#### **Training Graph**
📌 **Description**:
- Nodes represent **news articles**.
- Edges define **relationships** based on citations, similarities, or social interactions.

#### **Testing Graph**
![Testing Graph](<path-to-testing-graph-image>)
📌 **Description**:
- Nodes represent **unseen news articles**.
- The graph structure evaluates the **model’s generalization** capabilities.

---

### **Training and Test Loss Over Epochs**
![image](https://github.com/user-attachments/assets/77a7bfc6-b521-4ee3-8849-d49a460ef7de)

📉 **Observation**:
- **Training loss decreases** steadily, indicating learning.
- **Test loss fluctuates**, suggesting potential **overfitting** at certain epochs.

---

### **Metrics Over Epochs**
![image](https://github.com/user-attachments/assets/b67457a1-c848-4016-99bb-714d0fcf770c)

📊 **Observation**:
- Accuracy, Precision, Recall, and F1-score **improve over time**.
- Early fluctuations indicate **sensitivity to data variability**.

---

### **Confusion Matrix**
![image](https://github.com/user-attachments/assets/f38e9251-d3b1-47a8-aae2-66c17297b8a6)

📌 **Description**:
- **True Positives (1636)** and **True Negatives (1830)** indicate strong classification performance.
- **False Positives (80)** and **False Negatives (280)** highlight misclassifications.

---

### **ROC Curve**
![image](https://github.com/user-attachments/assets/5709e02d-4d1e-4103-af3e-4d36b1cfb717)

📌 **Observation**:
- **High ROC-AUC score** (close to 1) indicates strong classification capability.

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

## **3️⃣ Discussion and Insights**
### **Performance**
✅ **Key Takeaways**:
- **High ROC-AUC** score confirms effective classification.
- Model efficiently captures **relational information** from news articles.

### **Challenges**
⚠️ **Key Issues**:
- **Overfitting**: Test loss fluctuations suggest sensitivity to training data.
- **Sparse Graph Structures**: Test data graphs are often **less connected**, affecting model performance.

### **Future Work**
🚀 **Improvements**:
- **Enhanced Graph Features**: Include **social media trends, user credibility scores**.
- **Advanced GNN Architectures**: Experiment with **Graph Attention Networks (GAT)** and **GraphSAGE**.
- **Better Regularization**: Use **dropout layers** to mitigate overfitting.

---

## **4️⃣ How to Use**

### **1. Clone the Repository**
```bash
git clone <repository-url>
cd fake-news-detection-gnn
```

###  2. Install Dependencies
```bash
pip install torch torchvision torch_geometric networkx matplotlib seaborn pandas sklearn
```
### 3. Run the Training Script
```
python train.py
```
### 4. Evaluate the Model
```
python evaluate.py
```
## 5️⃣ Code Breakdown
### Graph Construction
- Uses NetworkX to create news article graphs.
- Converts PyTorch Geometric Data objects into NetworkX Graphs.
- 
### GNN Model Implementation
```
from torch_geometric.nn import GATConv
from torch.nn import Linear

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        # Graph Attention Layers
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)

        # Fully Connected Layers
        self.lin_news = Linear(in_channels, hidden_channels)
        self.lin0 = Linear(hidden_channels, hidden_channels)
        self.lin1 = Linear(2 * hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        h = self.conv3(h, edge_index).relu()

        h = gmp(h, batch)

        h = self.lin0(h).relu()

        root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
        root = torch.cat([root.new_zeros(1), root + 1], dim=0)
        news = x[root]
        news = self.lin_news(news).relu()

        out = self.lin1(torch.cat([h, news], dim=-1))
        return torch.sigmoid(out)
```
## Conclusion
This project successfully demonstrates fake news detection using Graph Neural Networks by leveraging news article relationships. 🚀

## 📌 Next Steps:

- Deploy as a real-time API.
- Enhance dataset with multi-source news.
- Optimize hyperparameters for improved performance.

