classdef Network < handle
    properties (SetAccess = protected)
        layers;
    end
    methods
        function obj = Network()
        end
        
        function AddLayer(obj, layer)
            if isempty(obj.layers)
                obj.layers = layer;
            else
                obj.layers(end+1) = layer;
            end
        end
        
        function Configure(obj, batch)
            for l = obj.layers
                l.Configure(batch);
                l.Forward(batch);
            end
        end
        
        function results = Apply(obj, batch)
            for l = obj.layers
                if nargout > 0
                    if ~exist('results', 'var')
                        results = copy(batch);
                    else
                        results(end+1) = copy(batch);%#ok
                    end
                end
                l.Forward(batch);    
            end
            if nargout > 0
                if ~exist('results', 'var')
                    results = copy(batch);
                else
                    results(end+1) = copy(batch);
                end
            end
        end
        
        function grads = Backprop(obj, results)
            delta = Batch(results(end).GetDataAsMatrix()-results(end).GetLabelsAsMatrix());
            grads = copy(delta);
            for i = length(results):-1:2
                obj.layers(i-1).Backward(results(i), delta);
                grads = [copy(delta), grads];%#ok
            end
        end
        
        function Update(obj, results, grads)
            for i = 1:length(results)-1
                obj.layers(i).Update(results(i), grads(i+1));
            end
        end
    end
end