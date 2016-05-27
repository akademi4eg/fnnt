classdef FullyConnectedLayer < Layer
    properties
        Transfer;
        DerTransfer;
        Weights;
        Biases;
        WeightsInitializer;
        ForwardFun;
        BackwardFun;
    end
    
    methods
        function obj = FullyConnectedLayer(neurons_num, transfer_fun, ...
                                       der_transfer_fun, w_init, b_init)
            if ~exist('w_init', 'var')
                w_init = @(n, m)0.1*randn(n, m);
            end
            if ~exist('b_init', 'var')
                b_init = @(n)0.1*randn(n, 1);
            end
            if ~exist('transfer_fun', 'var') || ~exist('der_transfer_fun', 'var')
                transfer_fun = @TansigTransfer;
                der_transfer_fun = @DerTansigTransfer;
            end
            obj.WeightsInitializer = @(x)w_init(neurons_num, x);
            obj.Biases = b_init(neurons_num);
            obj.Transfer = transfer_fun;
            obj.DerTransfer = der_transfer_fun;
        end
        
        function Configure(obj, batch)
            obj.Weights = obj.WeightsInitializer(batch.GetSampleWidth());
        end
        
        function fun = GetForwardFunction(obj)
            if isempty(obj.ForwardFun)
                obj.ForwardFun = @(x)obj.Transfer(bsxfun(@plus, obj.Weights*x, obj.Biases));
            end
            fun = obj.ForwardFun;
        end
        
        function fun = GetBackwardFunction(obj)
            if isempty(obj.BackwardFun)
                obj.BackwardFun = @(delta, y)(obj.Weights')*(obj.DerTransfer(y).*delta);
            end
            fun = obj.BackwardFun;
        end
        
        function Forward(obj, batch)
            batch.TransformData(obj.GetForwardFunction());
        end
        
        function Backward(obj, batch, grads_batch)
            gfun = obj.GetBackwardFunction();
            grads_batch.TransformData(@(x)gfun(x, batch.GetDataAsMatrix()));
        end
        
        function Update(obj, batch, grads_batch)
            obj.ForwardFun = [];
            obj.BackwardFun = [];
            dW = (grads_batch.GetDataAsMatrix()*batch.GetDataAsMatrix()');
            dW = dW / batch.GetBatchSize();
            db = mean(grads_batch.GetDataAsMatrix(), 2);
            obj.Weights = obj.Weights - 0.1 * dW;
            obj.Biases = obj.Biases - 0.1 * db;
        end
    end
end