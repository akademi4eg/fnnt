% This script trains a feed-forward neural network on one batch.
% Expects batch variable to be in workspace.
%% Create network
net = Network();
net.AddLayer(FullyConnectedLayer(100, @ReluTransfer, @DerReluTransfer));
net.AddLayer(SoftMaxLayer());
net.Configure(batches{1}.GetSample());
%% Add monitoring
net.AddTrainingPlot(@PlotMissclassError);
%% Start training
net.SetEpochsNum(1000);
net.SetEarlyStoping(10);
net.SetLoss(@CrossentropyLoss);
net.Train(batches);