classdef head < handle
    %HEAD Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        key, query, value
        inter, grad, batch
    end
    
    methods
        function obj = head(embd_dim, hidden_dim)
            %HEAD Construct an instance of this class
            %   Detailed explanation goes here
            obj.grad = struct;
            obj.inter = struct;
            obj.key = randn(embd_dim, hidden_dim) * (5/3)/power((embd_dim * hidden_dim), 0.5);
            obj.query = randn(embd_dim, hidden_dim) * (5/3)/power((embd_dim * hidden_dim), 0.5);
            obj.value = randn(embd_dim, hidden_dim) * (5/3)/power((embd_dim * hidden_dim), 0.5);
            obj.batch = 0;
        end
    
        function out = forward(obj, x)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            obj.inter.x = x;
            obj.inter.k = x * obj.key;
            obj.inter.q = x * obj.query;
            obj.inter.v = x * obj.value;
            obj.inter.C = size(x, 2);

            obj.inter.wei = obj.inter.q * transpose(obj.inter.k) / obj.inter.C;            
            %obj.inter.swei = softmax(obj.inter.wei, 2);
            
            %softmax
            obj.inter.swei = exp(obj.inter.wei - max(obj.inter.wei, [], 2));
            obj.inter.swei = obj.inter.swei ./ sum(obj.inter.swei, 2);

            out = obj.inter.swei * obj.inter.v;
        end

        function grad = backward(obj, dout)
            obj.batch = obj.batch + 1;
            dswei = dout * transpose(obj.inter.v);
            dv = transpose(dswei) * dout;
            dwei = zeros(size(obj.inter.wei));
            for i = 1:size(obj.inter.wei, 1)
                dwei(i, :) = dswei(i, :) * (-transpose(obj.inter.swei(i, :))*obj.inter.swei(i, :) + diag(obj.inter.swei(i, :)));
            end

            dk = transpose(dwei) * obj.inter.q / obj.inter.C;
            dq = dwei * obj.inter.k / obj.inter.C;
            
            if obj.batch == 1
                obj.grad.dvalue = transpose(obj.inter.x) * dv;
                obj.grad.dkey = transpose(obj.inter.x) * dk;
                obj.grad.dquery = transpose(obj.inter.x) * dq;
            else
                obj.grad.dvalue = obj.grad.dvalue + transpose(obj.inter.x) * dv;
                obj.grad.dkey = obj.grad.dkey + transpose(obj.inter.x) * dk;
                obj.grad.dquery = obj.grad.dquery + transpose(obj.inter.x) * dq;
            end

            grad = dv * transpose(obj.value) + dq * transpose(obj.query) + dk * transpose(obj.key);
        end

        function step(obj, lr)
            obj.key = obj.key - lr*obj.grad.dkey / obj.batch;
            obj.query = obj.query - lr*obj.grad.dquery / obj.batch;
            obj.value = obj.value - lr*obj.grad.dvalue / obj.batch;
        end

        function zero_grad(obj)
            obj.grad = struct;
            obj.batch = 0;
        end
    end
end

