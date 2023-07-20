from utils.prep import create_features
from utils.utility import (
    load_data_trn,
    load_dae_config,
    to_cuda,
    load_data_tst,
    save_weights,
    save_plot,
    save_report,
    create_results_folder,
    parse_arguments
)
from utils.train import train_autoencoder, train_softmax
from utils.test import load_model_weights, predict_softmax, evaluate_predictions

import torch


def main(index):
    for idx in range(index):
        create_results_folder(idx)
        dae_config          = load_dae_config("config/cnf_dae.csv")
        X_trn, Y_trn        = load_data_trn()
        X_trn               = to_cuda(torch.from_numpy(X_trn).float())
        Y_trn               = to_cuda(torch.from_numpy(Y_trn).long())
        A, encoder_weights  = train_autoencoder(X_trn, dae_config)
        softmax_weights     = train_softmax(A, Y_trn, dae_config)
        save_weights(
            encoder_weights, softmax_weights, idx
        )

        X_tst, Y_tst                     = load_data_tst()
        X_tst                            = to_cuda(torch.from_numpy(X_tst).float())
        encoder_weights, softmax_weights = load_model_weights(f"results/test{idx+1}/weights_test.pth")
        predictions                      = predict_softmax(X_tst, encoder_weights, softmax_weights)
        true_labels                      = Y_tst.argmax(axis=1)
        _, cm, report                    = evaluate_predictions(predictions, true_labels)
        save_plot(cm, idx)
        save_report(report, idx)

if __name__ == "__main__":
    args = parse_arguments()
    create_features()
    main(args.index)
