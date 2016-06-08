classdef (Abstract) Layer < handle
    methods (Abstract)
        Configure(obj, batch);
        Forward(obj, batch);
        Backward(obj, batch, grads_batch);
        Update(obj, batch_in, batch_out, grads_batch, train_params);
        reg = GetRegLoss(obj, train_params);
    end
end