classdef transformer < handle
    %TRANSFORMER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        pe, blk, W, b
        inter, grad, batch, dlogits
    end
    
    methods
        function obj = transformer(embd_dim, hidden_dim, num_heads, num_layers, block_dim, d_output)
            %TRANSFORMER Construct an instance of this class
            %   Detailed explanation goes here
            obj.grad = struct;
            obj.inter = struct;
            obj.inter.num_layers = num_layers;
            obj.pe = positionalEncoding(embd_dim, block_dim);
            obj.W = randn(embd_dim, d_output) * (5/3)/power((embd_dim * d_output), 0.5);
            obj.b = randn(1, d_output) * 0.1;

            obj.blk = block.empty(0, num_layers);
            for i = 1:num_layers
                obj.blk(i) = block(embd_dim, hidden_dim, num_heads);
            end
            obj.batch = 0;
        end

        function [y, loss] = predict(obj, x, target)
            [logits, loss] = obj.forward(x, target);
            logit_maxes = max(logits, [], 2, 'linear');
            logit_norm = logits - logit_maxes; %for numerical stability
            counts = exp(logit_norm);
            counts_sum = sum(counts, 2);
            counts_sum_inv = 1 ./ counts_sum;
            probs = counts .* counts_sum_inv;
            y = zeros(size(probs));
            for i = 1:size(probs, 1)
                y(i,:) = mnrnd(1,probs(i,:),1);
            end
        end
        
        function [logits, loss] = forward(obj, x, target)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            embds = x + obj.pe.forward(x);
            obj.inter.out{1} = obj.blk(1).forward(embds);

            for i = 2:obj.inter.num_layers
                obj.inter.out{i} = obj.blk(i).forward(obj.inter.out{i-1});
            end

            logits = obj.inter.out{obj.inter.num_layers} * obj.W + obj.b;
            [obj.dlogits, loss] = crossEntropy(logits, target);
        end

        function grad = backward(obj)
            obj.batch = obj.batch + 1;
            dout = obj.dlogits * transpose(obj.W);

            if (obj.batch == 1)
                obj.grad.dW = transpose(obj.inter.out{obj.inter.num_layers}) * obj.dlogits;
                obj.grad.db = sum(obj.dlogits, 1);
            else
                obj.grad.dW = obj.grad.dW + transpose(obj.inter.out{obj.inter.num_layers}) * obj.dlogits;
                obj.grad.db = obj.grad.db + sum(obj.dlogits, 1);
            end

            for i = obj.inter.num_layers:-1:1
                dout = obj.blk(i).backward(dout);
            end
            grad = dout;
        end

        function step(obj, lr)
            obj.W = obj.W - lr*obj.grad.dW / obj.batch;
            obj.b = obj.b - lr*obj.grad.db / obj.batch;

            for i = 1:obj.inter.num_layers
                obj.blk(i).step(lr);
            end
        end

        function zero_grad(obj)
            obj.grad = struct;
            obj.batch = 0;
            for i = 1:obj.inter.num_layers
                obj.blk(i).zero_grad();
            end
        end
    end
end

