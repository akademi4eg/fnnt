classdef NormalizerLayer < Layer
    properties (SetAccess = protected)
        Means;
        Stds;
        InvStds;
    end
    
    methods
        function obj = NormalizerLayer()
        end
        
        function Configure(~, ~)
        end
        
        function PreConfigure(obj, batches)
            TotalLen = 0;
            obj.Means = zeros(batches{1}.GetSampleWidth(), 1);
            obj.Stds = zeros(batches{1}.GetSampleWidth(), 1);
            for bi = batches
                obj.Means = obj.Means + sum(bi{1}.data, 2);
                TotalLen = TotalLen + bi{1}.GetBatchSize();
            end
            obj.Means = obj.Means/TotalLen;
            for bi = batches
                temp = bsxfun(@minus, bi{1}.data, obj.Means);
                obj.Stds = obj.Stds + sum(temp.^2, 2);
            end
            obj.Stds = sqrt(obj.Stds/TotalLen);
            obj.InvStds = 1./obj.Stds;
            obj.InvStds(isinf(obj.InvStds)) = 0;
        end
        
        function Forward(obj, batch)
            batch.TransformData(@(x)bsxfun(@times, bsxfun(@minus, x, obj.Means), obj.InvStds));
        end
        
        function Backward(obj, ~, grads_batch)
            grads_batch.TransformData(@(x)bsxfun(@plus, bsxfun(@times, x, obj.Stds), obj.Means));
        end
        
        function Update(~, ~, ~, ~, ~)
        end
        
        function reg = GetRegLoss(~, ~)
            reg = 0;
        end
    end
end