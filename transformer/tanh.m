classdef tanh < handle
    %TANH Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        inter
    end
    
    methods
        function obj = tanh()
            %TANH Construct an instance of this class
            %   Detailed explanation goes here
        end
        
        function y = forward(obj, x)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            obj.inter.y = tanh(x);
        end

        function grad = backward(obj, dy)
            grad = dy .* (1 - obj.inter.y .* obj.inter.y);
        end
    end
end

