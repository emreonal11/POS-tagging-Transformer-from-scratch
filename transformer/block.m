classdef block < handle
    %ENCODER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        MHA, W1, b1, W2, b2
        inter, grad, batch
    end
    
    methods
        function obj = block(embd_dim, hidden_dim, num_heads)
            %ENCODER Construct an instance of this class
            %   Detailed explanation goes here
            obj.grad = struct;
            obj.inter = struct;
            obj.MHA = multiHeadAttention(embd_dim, num_heads);
            obj.W1 = randn(embd_dim, hidden_dim) * (5/3)/power((embd_dim * hidden_dim), 0.5);
            obj.b1 = randn(1, hidden_dim) * 0.1;

            obj.W2 = randn(hidden_dim, embd_dim) * (5/3)/power((embd_dim * hidden_dim), 0.5);
            obj.b2 = randn(1, embd_dim) * 0.1;
            obj.batch = 0;
        end
        
        function out = forward(obj, x)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            obj.inter.attn_out = x + obj.MHA.forward(x);
            obj.inter.layer1 = obj.inter.attn_out * obj.W1 + obj.b1;
            obj.inter.activation1 = tanh(obj.inter.layer1);
            obj.inter.layer2 = obj.inter.activation1 * obj.W2 + obj.b2;
            out = obj.inter.attn_out + obj.inter.layer2;
        end

        function grad = backward(obj, dout)
            obj.batch = obj.batch + 1;
            dattn_out = dout;
            dlayer2 = dout;
            dactivation1 = dlayer2 * transpose(obj.W2);
            dlayer1 = dactivation1 .* (1 - obj.inter.activation1 .* obj.inter.activation1);
            dattn_out = dattn_out + dlayer1 * transpose(obj.W1);
            grad = dattn_out + obj.MHA.backward(dattn_out);

            if obj.batch == 1
                obj.grad.dW2 = transpose(obj.inter.activation1) * dlayer2;
                obj.grad.db2 = sum(dlayer2, 1);
    
                obj.grad.dW1 = transpose(obj.inter.attn_out) * dlayer1;
                obj.grad.db1 = sum(dlayer1, 1);
            else
                obj.grad.dW2 = obj.grad.dW2 + transpose(obj.inter.activation1) * dlayer2;
                obj.grad.db2 = obj.grad.db2 + sum(dlayer2, 1);
    
                obj.grad.dW1 = obj.grad.dW1 + transpose(obj.inter.attn_out) * dlayer1;
                obj.grad.db1 = obj.grad.db1 + sum(dlayer1, 1);
            end
        end

        function step(obj, lr)
            obj.W2 = obj.W2 - lr*obj.grad.dW2 / obj.batch;
            obj.b2 = obj.b2 - lr*obj.grad.db2 / obj.batch;
            obj.W1 = obj.W1 - lr*obj.grad.dW1 / obj.batch;
            obj.b1 = obj.b1 - lr*obj.grad.db1 / obj.batch;
            obj.MHA.step(lr);
        end

        function zero_grad(obj)
            obj.grad = struct;
            obj.batch = 0;
            obj.MHA.zero_grad();
        end
    end
end

