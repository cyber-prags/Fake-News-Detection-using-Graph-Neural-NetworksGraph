import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
)
import torch
import importlib.util

# Load your Python file dynamically
spec = importlib.util.spec_from_file_location("gnn_script", "fake_news_detection_using_gnn.py")
gnn_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gnn_script)

# Set up Streamlit App
st.title("Fake News Detection Using GNN")
st.write("This app provides insights and visualizations for fake news detection using a Graph Neural Network (GNN).")

# Sidebar Navigation
st.sidebar.title("Navigation")
options = ["Training Summary", "Metrics and Visualizations", "Prediction Analysis"]
choice = st.sidebar.radio("Go to:", options)

# Display content based on user selection
if choice == "Training Summary":
    st.header("Training Summary")
    st.write("Here are the training and testing loss trends over epochs.")
    
    # Fetch training results
    train_losses = gnn_script.train_losses
    test_losses = gnn_script.test_losses
    
    # Plot training vs test loss
    fig, ax = plt.subplots()
    ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker="o")
    ax.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss", marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Test Loss")
    ax.legend()
    st.pyplot(fig)

elif choice == "Metrics and Visualizations":
    st.header("Performance Metrics")
    st.write("Here are accuracy, precision, recall, F1 score, and ROC-AUC trends over epochs.")

    # Fetch metrics
    metrics = {
        "Accuracy": gnn_script.accuracies,
        "Precision": gnn_script.precisions,
        "Recall": gnn_script.recalls,
        "F1 Score": gnn_script.f1s,
        "ROC-AUC": gnn_script.roc_aucs,
    }

    # Plot metrics trends
    fig, ax = plt.subplots()
    for metric, values in metrics.items():
        ax.plot(range(1, len(values) + 1), values, label=metric, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric Value")
    ax.set_title("Performance Metrics Over Epochs")
    ax.legend()
    st.pyplot(fig)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    all_preds = gnn_script.all_preds
    all_labels = gnn_script.all_labels
    preds = torch.round(torch.cat(all_preds)).cpu().numpy()
    labels = torch.cat(all_labels).cpu().numpy()
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(labels, torch.cat(all_preds).cpu().numpy())
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC Curve", marker=".")
    ax.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

elif choice == "Prediction Analysis":
    st.header("Prediction Analysis")
    st.write("Analyze predictions, logits, and true labels.")

    # Display a sample prediction dataframe
    for data in gnn_script.test_loader:
        data = data.to(gnn_script.device)
        pred = gnn_script.model(data.x, data.edge_index, data.batch)
        df = pd.DataFrame()
        df["Logit"] = pred.detach().cpu().numpy()[:, 0]
        df["Prediction"] = torch.round(pred).detach().cpu().numpy()[:, 0]
        df["True Label"] = data.y.cpu().numpy()
        st.write(df.head(20))  # Display the first 20 rows
        break
