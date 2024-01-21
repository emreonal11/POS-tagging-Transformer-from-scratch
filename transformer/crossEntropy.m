function [dlogits, loss] = crossEntropy(logits, target)
%CROSSENTROPY Summary of this function goes here
%   Detailed explanation goes here
    [logit_maxes, i] = max(logits, [], 2, 'linear');
    logit_norm = logits - logit_maxes; %for numerical stability
    counts = exp(logit_norm);
    counts_sum = sum(counts, 2);
    counts_sum_inv = 1 ./ counts_sum;
    probs = counts .* counts_sum_inv;
    logprobs = log(probs);
    loss = mean(sum(-logprobs .* target, 2), 1);

    dlogprobs = -1.0 * target / (size(target, 1));
    dprobs = dlogprobs ./ probs;
    dcounts_sum_inv = sum((counts .* dprobs), 2); %this may be wrong
    dcounts = counts_sum_inv .* dprobs;
    dcounts_sum = -dcounts_sum_inv ./ (counts_sum .* counts_sum);
    dcounts = dcounts + ones(size(counts)) .* dcounts_sum;
    dnorm_logits = counts .* dcounts;
    dlogits = dnorm_logits;
    dlogit_maxes = sum(-dnorm_logits, 2);
    temp = zeros(size(logits));
    temp(i) = 1;
    dlogits = dlogits +  temp .* dlogit_maxes; %this may be wrong
end

