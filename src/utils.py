from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def metrics_scores(y_true, y_pred, pos_label):
    return {
        'accuracy': accuracy_score(y_true, y_pred, normalize=True),
        'precision': precision_score(y_true, y_pred, zero_division=0, pos_label=pos_label),
        'recall': recall_score(y_true, y_pred, zero_division=0, pos_label=pos_label),
        'f1': f1_score(y_true, y_pred, zero_division=0, pos_label=pos_label),
    }