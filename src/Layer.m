classdef (Abstract) Layer < matlab.mixin.Copyable
    methods (Abstract)
        Configure(obj, batch);
        PreTrain(obj, batches, train_params);
        Forward(obj, batch);
        Backward(obj, batch, grads_batch);
        train_params = Update(obj, batch_in, batch_out, grads_batch, train_params);
        reg = GetRegLoss(obj, train_params);
    end
end