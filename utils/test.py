from utils.utility import to_cuda

import torch
from sklearn.metrics import confusion_matrix, classification_report

def load_model_weights(file_path):
    weights_dict = torch.load(file_path)
    encoder_weights = [to_cuda(weights) for weights in weights_dict['encoder_weights']]
    softmax_weights = to_cuda(weights_dict['softmax_weights'])
    return encoder_weights, softmax_weights

def predict_softmax(X_tst, encoder_weights, softmax_weights):
    A = X_tst.clone()
    for weights in encoder_weights:
        A = torch.matmul(A, weights.t())

    predictions = torch.matmul(A, softmax_weights.t())
    return predictions

def evaluate_predictions(predictions, true_labels):
    predictions_cpu = predictions.cpu()
    predicted_labels = torch.argmax(predictions_cpu, dim=1).numpy()
    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    target_names = [f"Clase {i}" for i in range(len(set(true_labels)))]
    report = classification_report(true_labels, predicted_labels, target_names=target_names, zero_division=1)
    return predicted_labels, confusion_mat, report
