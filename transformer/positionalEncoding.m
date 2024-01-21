classdef positionalEncoding < handle
    %POSITIONALENCODING Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        pe
    end
    
    methods
        function obj = positionalEncoding(d_model, max_len)
            %POSITIONALENCODING Construct an instance of this class
            %   Detailed explanation goes here
            position = 0:(max_len-1);
            div_term = 0:2:(d_model-1);
            div_term = div_term * -log(10000) / d_model;
            obj.pe = zeros(max_len, d_model);
            obj.pe(:, 1:2:d_model) = sin(transpose(position) * div_term);
            obj.pe(:, 2:2:d_model) = cos(transpose(position) * div_term);
        end
        
        function out = forward(obj, x)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            sz = size(x);
            out = obj.pe(1:sz(1),:);
        end
    end
end

