# DeepReduce

In this work, we propose a novel appraoch for cost-effective testing of a deep learning model through input reduction. We present implementation code and all results in GitHub, and present all data (e.g., coverage data, trained models, modified models) in https://drive.google.com/file/d/1e5RlCxezVjg6E5Ny3B20uu5Z6vL6nku9/view?usp=sharing due to the limitation of file size.


Requirements:
    Python == 3.5.2
    Tensorflow == 1.11.0
    Keras == 2.2.4


All the implementations and results for the two datasets studied in our work are present in ./cifar-10 and ./mnist. In each studied model(i.e., NIN, VGG19, ResNet, Lenet1,Lenet4, and Lenet5), we present the implementation codes for training models, collecting NC, processing data, retraining modified models, our approach, and two baseline approaches. In order to reproduce our study, we give the guideline as follows:

Training models:
    Run the training code (e.g., in NIN, run 'python Network_in_Network_keras.py'). The trained model is saved as h5 file when the training process finishing.


Collecting NC:
    1) run 1cov.py to collect Neuron Coverage (e.g., run 'python 1cov.py 0.5' to collect NC0.5. paratemers can be 0.25, 0.5, and 0.75.)
    2) run 2process.py to handle data.
    3) run 3predict.py to collect the predict result for each testing input, which is used for our evaluation.
    4) run 4last_output.py to collect the outputs of neurons in the last hidden layer.

Our approach:
    run hsg-withcov.py with paratemers (e.g., run 'python hsg-withcov.py 0.5', paratemers can be 0.25, 0.5, and 0.75.). In order to collect all results with running our approach for one time, we collect the results of our appraoach for each iteration, and save them in ./result/DLR/. 'hsg-cov-xx.iteration.simplify_rawdata' and ''hsg-cov-xx.log.simplify_rawdata' present some parts results. 'hsg-cov-xx.iteration' and 'hsg-cov-xx.log' present the results with setting the termination condition to KL < 0.05, 0.01, 0.005, 0.001.


Two baseline approach:
    run 'bash fseklauto.sh' for the state-of-the-art approach, and run 'python random_kl.py' for random approach. The results are recorded in ./result/fse and ./result/RANDOM respectively.

Retraining modified models in regression scenario:
    1) the implementnation and results are presented in ./regression.
    2) some modification operations can not be applied to all models. For example, ADL/DEL/ADN/DEN can not be applied on VGG19, and LR/DR/MO can not be applied on LeNet.
