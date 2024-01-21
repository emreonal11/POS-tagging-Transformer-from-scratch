classdef multiHeadAttention < handle
    %MULTIHEADATTENTION Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        heads, W, b
        inter, grad, batch
    end
    
    methods
        function obj = multiHeadAttention(embd_dim, num_heads)
            %MULTIHEADATTENTION Construct an instance of this class
            %   Detailed explanation goes here
            obj.grad = struct;
            obj.inter = struct;
            obj.inter.embd_dim = embd_dim;
            obj.inter.num_heads = num_heads;
            obj.heads = head.empty(0, num_heads);
            for i = 1:num_heads
                obj.heads(i) = head(embd_dim, embd_dim / num_heads);
            end

            obj.W = randn(embd_dim, embd_dim) * (5/3)/power((embd_dim * embd_dim), 0.5);
            obj.b = randn(1, embd_dim) * 0.1;
            obj.batch = 0;
        end
        
        function out = forward(obj, x)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            obj.inter.attention = zeros(size(x));
            for i = 1:obj.inter.num_heads
                first = (i-1)*(obj.inter.embd_dim/obj.inter.num_heads)+1;
                last = i*(obj.inter.embd_dim/obj.inter.num_heads);

                obj.inter.attention(:,first:last) = obj.heads(i).forward(x);
            end

            out = obj.inter.attention * obj.W + obj.b;
        end

        function grad = backward(obj, dout)
            obj.batch = obj.batch + 1;
            if obj.batch == 1
                obj.grad.dW = transpose(obj.inter.attention) * dout;
                obj.grad.db = sum(dout, 1);
            else
                obj.grad.dW = obj.grad.dW + transpose(obj.inter.attention) * dout;
                obj.grad.db = obj.grad.db + sum(dout, 1);
            end

            dattention = dout * transpose(obj.W);
            grad = dout;

            for i = 1:obj.inter.num_heads
                first = (i-1)*(obj.inter.embd_dim/obj.inter.num_heads)+1;
                last = i*(obj.inter.embd_dim/obj.inter.num_heads);

                dout_attention = dattention(:,first:last);
                grad = grad + obj.heads(i).backward(dout_attention);
            end
        end

        function step(obj, lr)
            obj.W = obj.W - lr*obj.grad.dW / obj.batch;
            obj.b = obj.b - lr*obj.grad.db / obj.batch;
            for i = 1:obj.inter.num_heads
                obj.heads(i).step(lr);
            end
        end

        function zero_grad(obj)
            obj.grad = struct;
            obj.batch = 0;
            for i = 1:obj.inter.num_heads
                obj.heads(i).zero_grad();
            end
        end
    end
end

