train_data = readtable('data/train_data.csv');
valid_data = readtable('data/valid_data.csv');
test_data = readtable('data/test_data.csv');
embeddings = readtable('data/wv.csv');

cleaned_embeddings = cellfun(@(str_embeddings) str2num(regexprep(strrep(strrep(str_embeddings, '[', ''), ']', ''), '\s+', ' ')), embeddings.vectors, 'UniformOutput', false);
word2embedding = containers.Map(embeddings.word, cleaned_embeddings);

[X_train, X_test, X_valid, Y_train, Y_valid, Y_test, train_padding, valid_padding, test_padding] = prep_data(train_data, valid_data, test_data, word2embedding);
model = transformer(64, 256, 2, 2, 10, 4);

batch_size = 32;

loss_xcor = [];
loss_ycor = [];

for i = 1:10000
    if rem(i, 100) == 0
        [Xb, yb] = get_batch(X_valid, Y_valid);
        [y, loss] = model.predict(Xb, yb);
        correct = sum(sum(y .* yb, 2));
        total = size(yb, 1);
        fprintf('Iteration = %d Validation Loss = %f Validation Accuracy = %f \n', i, loss, correct/total);
    end

    model.zero_grad();
    batch_loss = 0;
    for j = 1:batch_size
        [Xb, yb] = get_batch(X_train, Y_train);
        [~, loss] = model.forward(Xb, yb);
        batch_loss = batch_loss + loss;
        model.backward();
    end
    model.step(0.001);
    loss_xcor = [loss_xcor, i];
    loss_ycor = [loss_ycor, batch_loss];

    %fprintf('Iteration = %d Training Loss = %f\n', i, batch_loss/batch_size);
end

plot(loss_xcor, loss_ycor);

correct = 0;
total = 0;

for i = 1:size(X_test,1)
    Xb = cell2mat(X_test(i, :));
    yb = cell2mat(Y_test(i, :));
    [y, loss] = model.predict(Xb, yb);
    correct = correct + sum(sum(y .* yb, 2));
    total = total + size(yb, 1);
    fprintf('Testing Accuracy = %f\n', correct/total);
end

fprintf('Testing Accuracy = %f\n', correct/total);
