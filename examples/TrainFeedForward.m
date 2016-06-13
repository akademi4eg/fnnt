% This script trains a feed-forward neural network.
% Expects batch variable to be in workspace.
%% Create network
net = Network();
norm = NormalizerLayer();
net.AddLayer(norm);
net.AddLayer(DropoutLayer(0.2));
net.AddLayer(FullyConnectedLayer(1000, @ReluTransfer, @DerReluTransfer));
net.AddLayer(DropoutLayer(0.5));
net.AddLayer(FullyConnectedLayer(1200, @ReluTransfer, @DerReluTransfer));
net.AddLayer(DropoutLayer(0.5));
net.AddLayer(FullyConnectedLayer(800, @ReluTransfer, @DerReluTransfer));
net.AddLayer(DropoutLayer(0.25));
net.AddLayer(FullyConnectedLayer(100, @ReluTransfer, @DerReluTransfer));
net.AddLayer(DropoutLayer(0.1));
net.AddLayer(SoftMaxLayer());
norm.PreConfigure(batches);
net.Configure(batches{1}.GetSample());
%% Add monitoring
net.AddTrainingPlot(@PlotMissclassError);
%% Training. Phase 1
% use momentum, dropout and regularization
net.SetMomentum('CM', 0.5, 0.001, 0.9);
net.SetRegularization('L1', 0.1);
net.SetTrainParams(1000, 0.1, 0.99);
net.SetEarlyStoping(15);
net.SetLoss(@CrossentropyLoss);
net.Train(batches);
%% Training. Phase 2
% remove dropout for input layer, reduce regularization
net.RemoveLayers(1);
net.SetRegularization('L1', 0.01);
net.Train(batches);
%% Training. Phase 3
% completely remove dropout, disable regularization, add crossvalidation to
% train set, train for 100 epochs without early stopping
for bi = batches
    bi{1}.SetBatchType('trn');
end
net.RemoveLayers('DropoutLayer');
net.SetRegularization('none');
net.SetTrainParams(100);
net.SetEarlyStoping(Inf);
net.Train(batches);