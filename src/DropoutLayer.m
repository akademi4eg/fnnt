classdef DropoutLayer < Layer
    properties (SetAccess = protected)
        Mask;
        DropRate;
    end
    
    methods
        function obj = DropoutLayer(prob)
            obj.DropRate = prob;
        end
        
        function Configure(obj, batch)
            obj.Mask = true(batch.GetSampleWidth(), 1);
        end
        
        function PreTrain(~, ~, ~)
        end
        
        function Forward(obj, batch)
            out_sum = mean(obj.Mask);
            batch.TransformData(@(x)bsxfun(@times, x, obj.Mask/out_sum));
        end
        
        function Backward(obj, ~, grads_batch)
            grads_batch.TransformData(@(x)bsxfun(@times, x, obj.Mask));
        end
        
        function train_params = Update(obj, ~, ~, ~, train_params)
            obj.Mask = rand(size(obj.Mask)) > obj.DropRate;
            if ~any(obj.Mask)
                obj.Mask(randi(length(obj.Mask))) = true;
            end
        end
        
        function reg = GetRegLoss(~, ~)
            reg = 0;
        end
    end
end