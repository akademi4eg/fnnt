% This script trains a feed-forward neural network on one batch.
% Expects batch variable to be in workspace.
%% Create network
net = Network();
net.AddLayer(FullyConnectedLayer(100));
net.AddLayer(FullyConnectedLayer(10));
net.Configure(batch.GetSample());
%% Add monitoring
net.AddTrainingPlot(@PlotMSE);
%% Start training
net.SetEpochsNum(50);
net.Train(batch);