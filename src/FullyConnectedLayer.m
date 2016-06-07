classdef FullyConnectedLayer < Layer
    properties (SetAccess = protected)
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
            if nargin < 1 || ~isnumeric(neurons_num)
                % prevent execution for subclasses like softmax
                return;
            end
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
        
        function reg = GetRegLoss(obj, train_params)
            if strcmp(train_params.regularization.type, 'L2')
                reg = train_params.regularization.param * mean(obj.Weights(:).^2);
            elseif strcmp(train_params.regularization.type, 'L1')
                reg = train_params.regularization.param * mean(abs(obj.Weights(:)));
            else
                reg = 0;
            end
        end
        
        function Update(obj, batch_in, batch_out, grads_batch, train_params)
            obj.ForwardFun = [];
            obj.BackwardFun = [];
            dW = (grads_batch.GetDataAsMatrix().*obj.DerTransfer(batch_out.GetDataAsMatrix()))*batch_in.GetDataAsMatrix()';
            dW = dW / batch_in.GetBatchSize();
            if strcmp(train_params.regularization.type, 'L2')
                dW = dW + train_params.regularization.param * obj.Weights/numel(obj.Weights);
            elseif strcmp(train_params.regularization.type, 'L1')
                dW = dW + train_params.regularization.param * sign(obj.Weights)/numel(obj.Weights);
            end
            db = mean(grads_batch.GetDataAsMatrix(), 2);
            obj.Weights = obj.Weights - train_params.learn_rate * dW;
            obj.Biases = obj.Biases - train_params.learn_rate * db;
        end
    end
end