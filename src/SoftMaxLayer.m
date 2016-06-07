classdef SoftMaxLayer < FullyConnectedLayer
    methods
        function obj = SoftMaxLayer(w_init)
            if ~exist('w_init', 'var')
                w_init = @(n, m)0.1*randn(n, m);
            end
            obj.WeightsInitializer = w_init;
            obj.Biases = [];
            obj.Transfer = @(x)bsxfun(@times, exp(x), 1./sum(exp(x), 1));
            obj.DerTransfer = @(x)x.*(1-x);
        end
        
        function Configure(obj, batch)
            obj.Weights = obj.WeightsInitializer(batch.GetLabelsNum(), batch.GetSampleWidth());
        end
        
        function fun = GetForwardFunction(obj)
            if isempty(obj.ForwardFun)
                obj.ForwardFun = @(x)obj.Transfer(obj.Weights*x);
            end
            fun = obj.ForwardFun;
        end
        
        function Update(obj, batch_in, batch_out, grads_batch)
            obj.ForwardFun = [];
            obj.BackwardFun = [];
            dW = (grads_batch.GetDataAsMatrix().*obj.DerTransfer(batch_out.GetDataAsMatrix()))*batch_in.GetDataAsMatrix()';
            dW = dW / batch_in.GetBatchSize();
            obj.Weights = obj.Weights - 0.1 * dW;
        end
    end
end