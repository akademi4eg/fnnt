classdef (Abstract) Layer < handle
    methods (Abstract)
        Configure(obj, batch);
        Forward(obj, batch);
        fun = GetForwardFunction(obj);
        Backward(obj, batch, grads_batch);
        fun = GetBackwardFunction(obj);
        Update(obj, batch, grads_batch);
    end
end