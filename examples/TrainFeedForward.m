% This script trains a feed-forward neural network on one batch.
% Expects batch variable to be in workspace.
%% Create network
net = Network();
net.AddLayer(DropoutLayer(0.2));
net.AddLayer(FullyConnectedLayer(1000, @ReluTransfer, @DerReluTransfer));
net.AddLayer(DropoutLayer(0.5));
net.AddLayer(FullyConnectedLayer(1000, @ReluTransfer, @DerReluTransfer));
net.AddLayer(DropoutLayer(0.5));
net.AddLayer(SoftMaxLayer());
net.Configure(batches{1}.GetSample());
%% Add monitoring
net.AddTrainingPlot(@PlotMissclassError);
%% Start training
net.SetMomentum('CM', 0.5, 0.001, 0.9);
net.SetTrainParams(1000, 0.01);
net.SetEarlyStoping(10);
net.SetLoss(@CrossentropyLoss);
net.Train(batches);