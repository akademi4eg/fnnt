classdef NormalizerLayer < Layer
    properties (SetAccess = protected)
        Mode = 'z-score';
        Means = [];
        Stds = [];
        Maxs = [];
        Mins = [];
        InvStds = [];
    end
    
    methods
        function obj = NormalizerLayer(mode)
            obj.Mode = mode;
        end
        
        function Configure(~, ~)
        end
        
        function PreTrain(obj, batches, ~)
            if strcmp(obj.Mode, 'z-score')
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
            elseif strcmp(obj.Mode, 'binary')
                obj.Mins = zeros(batches{1}.GetSampleWidth(), 1);
                obj.Maxs = zeros(batches{1}.GetSampleWidth(), 1);
                for bi = batches
                    obj.Mins = min(obj.Mins, min(bi{1}.data, [], 2));
                    obj.Maxs = max(obj.Maxs, max(bi{1}.data, [], 2));
                end
                obj.Stds = obj.Maxs-obj.Mins;
                obj.InvStds = 1./obj.Stds;
                obj.InvStds(isinf(obj.InvStds)) = 0;
            end
        end
        
        function Forward(obj, batch)
            if isempty(obj.InvStds), return; end
            if strcmp(obj.Mode, 'z-score')
                batch.TransformData(@(x)bsxfun(@times, bsxfun(@minus, x, obj.Means), obj.InvStds));
            elseif strcmp(obj.Mode, 'binary')
                batch.TransformData(@(x)bsxfun(@times, bsxfun(@minus, x, obj.Mins), obj.InvStds));
            end
        end
        
        function Backward(obj, ~, grads_batch)
            if strcmp(obj.Mode, 'z-score')
                grads_batch.TransformData(@(x)bsxfun(@plus, bsxfun(@times, x, obj.Stds), obj.Means));
            elseif strcmp(obj.Mode, 'binary')
                grads_batch.TransformData(@(x)bsxfun(@plus, bsxfun(@times, x, obj.Stds), obj.Mins));
            end
        end
        
        function train_params = Update(~, ~, ~, ~, train_params)
        end
        
        function reg = GetRegLoss(~, ~)
            reg = 0;
        end
    end
end