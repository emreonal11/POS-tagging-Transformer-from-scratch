function [train_embs, test_embs, valid_embs, Y_train, Y_valid, Y_test, train_padding, valid_padding, test_padding] = prep_data(train_data, valid_data, test_data, word2embedding)
    [train_embs, train_labels, SEQ_LEN] = embed(train_data, word2embedding, 'train');
    [valid_embs, valid_labels, ~] = embed(valid_data, word2embedding, 'valid');
    [test_embs, test_labels, ~] = embed(test_data, word2embedding, 'test');
    
    [train_embs, train_padding] = pad(train_embs, SEQ_LEN);
    [test_embs, test_padding] = pad(test_embs, SEQ_LEN);
    [valid_embs, valid_padding] = pad(valid_embs, SEQ_LEN);

    positional_encoding = get_positional_encoding(SEQ_LEN, 64);
    train_embs = encode_positions(train_embs, positional_encoding);
    test_embs = encode_positions(test_embs, positional_encoding);
    valid_embs = encode_positions(valid_embs, positional_encoding);

    train_labels = cellfun(@(str_labels) str2num(str_labels), train_labels, 'UniformOutput', false);
    valid_labels = cellfun(@(str_labels) str2num(str_labels), valid_labels, 'UniformOutput', false);
    test_labels = cellfun(@(str_labels) str2num(str_labels), test_labels, 'UniformOutput', false);
    train_labels = convert_tags(train_labels);
    valid_labels = convert_tags(valid_labels);
    test_labels = convert_tags(test_labels);

    [Y_train, Y_valid, Y_test] = one_hot(train_labels, valid_labels, test_labels, SEQ_LEN);

    [train_embs, Y_train] = chunk_data(train_embs, Y_train, train_padding);
    [valid_embs, Y_valid] = chunk_data(valid_embs, Y_valid, valid_padding);
    [test_embs, Y_test] = chunk_data(test_embs, Y_test, test_padding);

end

function [Y_train, Y_valid, Y_test] = one_hot(train_labels, valid_labels, test_labels, SEQ_LEN)
    Y_train = cell(size(train_labels));
    Y_test = cell(size(test_labels));
    Y_valid = cell(size(valid_labels));
    
    for i = 1:length(train_labels)
        sample_labels = zeros(SEQ_LEN, 4);
        
        for j = 1:length(train_labels{i})
            sample_labels(j, train_labels{i}(j)) = 1;
        end
        Y_train{i} = sample_labels;
    end
    
    for i = 1:length(test_labels)
        sample_labels = zeros(SEQ_LEN, 4);
        
        for j = 1:length(test_labels{i})
            sample_labels(j, test_labels{i}(j)) = 1;
        end
        Y_test{i} = sample_labels;
    end
    
    for i = 1:length(valid_labels)
        sample_labels = zeros(SEQ_LEN, 4);
        
        for j = 1:length(valid_labels{i})
            sample_labels(j, valid_labels{i}(j)) = 1;
        end
        Y_valid{i} = sample_labels;
    end
end


function new_tags = convert_tags(old_tags)
    noun_tags = [21, 24, 22, 23, 25, 28, 29];
    verb_tags = [37, 38, 39, 40, 41, 42];
    adj_adv_tags = [16, 17, 18, 30, 31, 32];
    % default is 4 (other)
    % noun (1), verb (2), and adjective/adverb (3)
    
    new_tags = cell(size(old_tags));
    for i = 1:length(old_tags)
        old_tags_i = old_tags{i};
        new_tags_i = 4 * ones(1, length(old_tags_i));

        new_tags_i(ismember(old_tags_i, noun_tags)) = 1;
        new_tags_i(ismember(old_tags_i, verb_tags)) = 2;
        new_tags_i(ismember(old_tags_i, adj_adv_tags)) = 3;

        new_tags{i} = new_tags_i;
    end
end

function [unpadded_embs, labels, max_len] = embed(data, word2embedding, split)
    fprintf('embedding %s... \n', split);
    unpadded_embs = cell(size(data.tokens));
    valid_samples = true(length(unpadded_embs), 1);
    n_invalid=0;
    for i = 1:length(unpadded_embs)
        tokens = data.tokens{i};
        words = cellfun(@(x) x(2:end-1), strsplit(tokens(2:end-1), ', '), 'UniformOutput', false);
        
        sentence_emb = cell(size(words));
        for j = 1:length(words)
            word = words{j};
    
            if isKey(word2embedding, word)
                sentence_emb{j} = word2embedding(word);
            else
                %valid_samples(i) = false;
                sentence_emb{j} = zeros(1,64);
                n_invalid = n_invalid + 1;
            end
        end
        
        if valid_samples(i)
            unpadded_embs{i} = cell2mat(sentence_emb');
        end
    end
    fprintf('Found %d unknown embeddings \n', n_invalid);
    old_len = length(unpadded_embs);
    unpadded_embs = unpadded_embs(valid_samples);
    fprintf('Removed %d / %d samples \n', old_len - length(unpadded_embs), old_len);
    labels = data.pos_tags(valid_samples);

    max_len = max(cellfun(@(sentence) size(sentence, 1), unpadded_embs));
end

function [padded_embs, padding_mask] = pad(unpadded_embs, SEQ_LEN)
    padded_embs = cell(size(unpadded_embs));
    padding_mask = cell(size(unpadded_embs));

    for i = 1:length(unpadded_embs)
        sentence = unpadded_embs{i};
        current_length = size(sentence, 1);

        if current_length < SEQ_LEN
            padding_length = SEQ_LEN - current_length;
            zero_padding = zeros(padding_length, size(sentence, 2));
            padded_embs{i} = [sentence; zero_padding];
            padding_mask{i} = [ones(current_length, 1); zeros(padding_length, 1)];
        else
            padded_embs{i} = sentence(1:SEQ_LEN, :);
            padding_mask{i} = ones(SEQ_LEN, 1);
        end
    end
end

function positional_encoding = get_positional_encoding(SEQ_LEN, DIM)
    positional_encoding = zeros(SEQ_LEN, DIM);
    
    for pos = 1:SEQ_LEN
        for i = 1:DIM
            if mod(i,2) == 0
                % even index -> cos (flipped sin/cos because 1-indexing)
                positional_encoding(pos, i) = cos(pos / (10000^((i-2)/DIM)));
            else
                % odd index -> sin
                positional_encoding(pos, i) = sin(pos / (10000^((i-1)/DIM)));
            end
        end
    end
end

function encoded_vecs = encode_positions(vecs, positional_encoding)
    encoded_vecs = cell(size(vecs));
    for i = 1:length(vecs)
        encoded_vecs{i} = vecs{i} + positional_encoding;
    end
end

function [X_chunked, Y_chunked] = chunk_data(X_old, Y_old, old_padding)
    chunk_size = 10;
    X_chunked = {};
    Y_chunked = {};
    
    for i = 1:length(X_old)
        sentence = X_old{i};
        padding = old_padding{i};
        labels = Y_old{i};
    
        num_tokens = sum(padding);
        if num_tokens >= chunk_size
            num_chunks = floor(num_tokens / chunk_size);
    
            for j = 1:num_chunks
                start_idx = (j-1) * chunk_size + 1;
                end_idx = j*chunk_size;
    
                tokens_chunk = sentence(start_idx:end_idx, :);
                labels_chunk = labels(start_idx:end_idx, :);
    
                X_chunked{end+1, 1} = tokens_chunk;
                Y_chunked{end+1, 1} = labels_chunk;
            end
        end
    end
end
