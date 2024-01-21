classdef softmax < handle
    %SOFTMAX Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        inter
    end
    
    methods
        function obj = softmax()
            %SOFTMAX Construct an instance of this class
            %   Detailed explanation goes here
        end
        
        function y = forward(obj, x)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            obj.inter.y = exp(x - max(x, [], 2));
            y = obj.inter.y ./ sum(obj.inter.y, 2);
            obj.inter.y = y;
        end

        function grad = backward(obj, dy)
            grad = zeros(size(obj.inter.y));
            for i = 1:size(obj.inter.y, 1)
                grad(i, :) = dy(i, :) * (-transpose(obj.inter.y(i, :)) * obj.inter.y(i, :) + diag(obj.inter.y(i, :)));
            end
        end
    end
end

