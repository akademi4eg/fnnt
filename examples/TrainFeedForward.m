% This script trains a feed-forward neural network.
% Expects batch variable to be in workspace.
%% Load data
load ../data/mnist.mat
%% Create network
net = Network();
net.AddLayer(NormalizerLayer('binary'));
% net.AddLayer(DropoutLayer(0.2));
net.AddLayer(FullyConnectedLayer(10, @LogisticTransfer, @DerLogisticTransfer));
% net.AddLayer(DropoutLayer(0.5));
net.AddLayer(FullyConnectedLayer(10, @LogisticTransfer, @DerLogisticTransfer));
% net.AddLayer(DropoutLayer(0.5));
net.AddLayer(SoftMaxLayer());
net.Configure(batches{1}.GetSample());
%% Add monitoring
net.SetMonitoringMode('text');
% net.AddTrainingPlot(@PlotMissclassError);
%% Pretraining
net.SetTrainParams(0, 0.00001);
net.SetRegularization('L2', 0.1);
net.SetEarlyStoping(10);
net.PreTrain(batches);

load('../data/mnist.mat', 'batches');
%% Training. Phase 1
% use momentum, dropout and regularization
net.SetMomentum('CM', 0.5, 0.005, 0.9);
net.SetRegularization('L2', 0.0001);
net.SetTrainParams(100, 5, 0.8);
net.SetEarlyStoping(15);
net.SetLoss(@CrossentropyLoss);
net.Train(batches);

out = copy(test_batch);
net.Apply(out);
miss = GetMissclassRate(out);
fprintf('After phase 1 missclass rate: %2.2f%%.\n', miss);
% %% Training. Phase 2
% % remove dropout for input layer, reduce regularization
% net.RemoveLayers(1);
% net.SetRegularization('L1', 0.01);
% net.Train(batches);
% 
% out = copy(test_batch);
% net.Apply(out);
% miss = GetMissclassRate(out);
% fprintf('After phase 2 missclass rate: %2.2f%%.\n', miss);
% %% Training. Phase 3
% % completely remove dropout, disable regularization, add crossvalidation to
% % train set, train for 100 epochs without early stopping
% for bi = batches
%     bi{1}.SetBatchType('trn');
% end
% net.RemoveLayers('DropoutLayer');
% net.SetRegularization('none');
% net.SetTrainParams(50);
% net.SetEarlyStoping(Inf);
% net.Train(batches);
% 
% out = copy(test_batch);
% net.Apply(out);
% miss = GetMissclassRate(out);
% fprintf('After phase 3 missclass rate: %2.2f%%.\n', miss);