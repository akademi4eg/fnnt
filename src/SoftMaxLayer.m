classdef SoftMaxLayer < FullyConnectedLayer
    methods
        function obj = SoftMaxLayer(w_init)
            if ~exist('w_init', 'var')
                w_init = @(n, m)0.01*randn(n, m);
            end
            obj.WeightsInitializer = w_init;
            obj.Biases = [];
            obj.Transfer = @SoftMaxTransfer;
            obj.DerTransfer = @DerSoftMaxTransfer;
        end
        
        function Configure(obj, batch)
            obj.Weights = obj.WeightsInitializer(batch.GetLabelsNum(), batch.GetSampleWidth());
        end
        
        function PreTrain(~, ~, ~)
        end
        
        function fun = GetForwardFunction(obj)
            if isempty(obj.ForwardFun)
                obj.ForwardFun = @(x)obj.Transfer(obj.Weights*x);
            end
            fun = obj.ForwardFun;
        end
        
        function Update(obj, batch_in, batch_out, grads_batch, train_params)
            obj.ForwardFun = [];
            obj.BackwardFun = [];
            % weights delta
            dW = (grads_batch.GetDataAsMatrix().*obj.DerTransfer(batch_out.GetDataAsMatrix()))*batch_in.GetDataAsMatrix()';
            dW = dW / batch_in.GetBatchSize();
            % regularization
            if strcmp(train_params.regularization.type, 'L2')
                dW = dW + train_params.regularization.param * obj.Weights/numel(obj.Weights);
            elseif strcmp(train_params.regularization.type, 'L1')
                dW = dW + train_params.regularization.param * sign(obj.Weights)/numel(obj.Weights);
            end
            if max(abs(dW(:))) > train_params.max_delta/train_params.learn_rate.value
                dW = dW/max(abs(dW(:)))*train_params.max_delta/train_params.learn_rate.value;
            end
            % momentum
            if strcmp(train_params.momentum.type, 'CM')
                if isempty(obj.DeltaWeightsMom)
                    obj.DeltaWeightsMom = train_params.learn_rate.value*dW;
                else
                    obj.DeltaWeightsMom = train_params.momentum.param*obj.DeltaWeightsMom ...
                        + (1-train_params.momentum.param)*train_params.learn_rate.value*dW;
                end
                dW = obj.DeltaWeightsMom;
            else
                dW = train_params.learn_rate.value * dW;
            end
            % update!
            obj.Weights = obj.Weights - dW;
            
            obj.ForwardFun = [];
            obj.BackwardFun = [];
        end
    end
end