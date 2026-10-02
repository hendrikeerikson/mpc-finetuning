import shared3p;
import shared3p_matrix;
import stdlib;
import profiling;

import common;

domain pd_shared3p shared3p;


void main() {
    // Network dimensions: input -> hidden1 -> hidden2 -> output.
    uint input_dim = 784;
    uint hidden1_dim = 128;
    uint hidden2_dim = 64;
    uint output_dim = 10;
    uint bs = 64;  // batch size
    uint epoch = 1;
    uint n_batch = 5;
    float32 lr = 0.001;
    bool printBatchLoss = true;

    Data<pd_shared3p, float32> train_x = readArgData("x_train", 11800::uint, 784::uint);
    Data<pd_shared3p, float32> train_y = readArgData("y_train", 11800::uint, 2::uint);
    Data<pd_shared3p, float32> test_x = readArgData("x_test", 100::uint, 784::uint);
    Data<pd_shared3p, float32> test_y = readArgData("y_test", 100::uint, 2::uint);
    pd_shared3p float32[[2]] XY(train_x.nrow, train_x.ncol + train_y.ncol);
    XY[:, :train_x.ncol] = train_x.data;
    XY[:, train_x.ncol:] = train_y.data;

    pd_shared3p float32[[2]] batch_x(bs, train_x.ncol);
    pd_shared3p float32[[2]] batch_y(bs, train_y.ncol);

    // Adam hyperparameters
    pd_shared3p float32 beta1 = 0.9;
    pd_shared3p float32 beta2 = 0.999;
    pd_shared3p float32 eps = 0.0001;
    pd_shared3p float32 beta1_pow = 1.0;
    pd_shared3p float32 beta2_pow = 1.0;

    // Train every layer from scratch using the existing initialization helpers.
    pd_shared3p float32[[2]] W1 = initWeights(hidden1_dim, input_dim);
    pd_shared3p float32[[1]] b1 = initBias(hidden1_dim);
    pd_shared3p float32[[2]] W2 = initWeights(hidden2_dim, hidden1_dim);
    pd_shared3p float32[[1]] b2 = initBias(hidden2_dim);
    pd_shared3p float32[[2]] W3 = initWeights(output_dim, hidden2_dim);
    pd_shared3p float32[[1]] b3 = initBias(output_dim);

    pd_shared3p float32[[2]] zh1(bs, shape(W1)[0]);
    pd_shared3p bool[[2]] mask1(bs, shape(W1)[0]);
    pd_shared3p float32[[2]] zh2(bs, shape(W2)[0]);
    pd_shared3p bool[[2]] mask2(bs, shape(W2)[0]);
    pd_shared3p float32[[2]] logitsProb(bs, output_dim);
    pd_shared3p float32[[2]] delta(bs, output_dim);
    pd_shared3p float32 loss;
    pd_shared3p float32[[2]] grad_W3(output_dim, shape(W2)[0]);
    pd_shared3p float32[[1]] grad_b3(output_dim);
    pd_shared3p float32[[2]] grad_zh2(bs, shape(W2)[0]);
    pd_shared3p float32[[2]] grad_W2(shape(W2)[0], shape(W1)[0]);
    pd_shared3p float32[[1]] grad_b2(size(b2));
    pd_shared3p float32[[2]] grad_zh1(bs, shape(W1)[0]);
    pd_shared3p float32[[2]] grad_W1(shape(W1)[0], shape(W1)[1]);
    pd_shared3p float32[[1]] grad_b1(size(b1));

    // Adam state: concatenate W1, b1, W2, b2, W3, b3.
    uint adam_end_W1 = 0 + size(W1);
    uint adam_end_b1 = adam_end_W1 + size(b1);
    uint adam_end_W2 = adam_end_b1 + size(W2);
    uint adam_end_b2 = adam_end_W2 + size(b2);
    uint adam_end_W3 = adam_end_b2 + size(W3);
    uint adam_end_b3 = adam_end_W3 + size(b3);
    pd_shared3p float32[[1]] adam_params(adam_end_b3);
    pd_shared3p float32[[1]] adam_grad(adam_end_b3);
    pd_shared3p float32[[1]] adam_m(adam_end_b3);
    pd_shared3p float32[[1]] adam_v(adam_end_b3);

    // Pack initial parameters once; retain the packed state across steps.
    adam_params[0:adam_end_W1] = reshape(W1, size(W1));
    adam_params[adam_end_W1:adam_end_b1] = b1;
    adam_params[adam_end_b1:adam_end_W2] = reshape(W2, size(W2));
    adam_params[adam_end_W2:adam_end_b2] = b2;
    adam_params[adam_end_b2:adam_end_W3] = reshape(W3, size(W3));
    adam_params[adam_end_W3:adam_end_b3] = b3;

    pd_shared3p float32 step_scale;

    pd_shared3p float32 loss_sum = 0.0;
    pd_shared3p float32 eps_scaled;


    for (uint i=0; i< epoch; i++) {
        print("epoch: ", i+1, "/", epoch);

        for (uint j=0; j<n_batch; j++) {
            if (printBatchLoss) print("- batch: ", j+1, "/", n_batch);

            uint32 sec_batch = section("batch");
            batch_x = XY[j * bs: (j+1) * bs, :input_dim];
            batch_y = XY[j * bs: (j+1) * bs, input_dim:];


            uint32 sec= section("forward");
            // forward
            zh1 = addBias(matrixMultiplication(batch_x, transpose(W1)), b1);
            mask1 = reLUMask(zh1);
            zh1 = reLU(zh1);
            zh2 = addBias(matrixMultiplication(zh1, transpose(W2)), b2);
            mask2 = reLUMask(zh2);
            zh2 = reLU(zh2);
            logitsProb = addBias(matrixMultiplication(zh2, transpose(W3)), b3);

            endSection(sec);
            sec = section("loss");

            logitsProb = softmax(logitsProb);
            loss = crossEntropyError(batch_y, logitsProb);
            delta = crossEntropyDelta(batch_y, logitsProb);
            loss_sum += loss;
            if (printBatchLoss) print(" Loss: ", declassify(loss));
            endSection(sec);

            // backward
            sec = section("backward");
            grad_W3 = matrixMultiplication(transpose(delta), zh2);
            grad_b3 = colSums(delta);
            grad_zh2 = matrixMultiplication(delta, W3);
            grad_zh2 *= (float32) mask2;
            grad_W2 = matrixMultiplication(transpose(grad_zh2), zh1);
            grad_b2 = colSums(grad_zh2);
            grad_zh1 = matrixMultiplication(grad_zh2, W2);
            grad_zh1 *= (float32) mask1;
            grad_W1 = matrixMultiplication(transpose(grad_zh1), batch_x);
            grad_b1 = colSums(grad_zh1);
            endSection(sec);

            // Adam
            sec = section("adam");
            adam_grad[0:adam_end_W1] = reshape(grad_W1, size(grad_W1));
            adam_grad[adam_end_W1:adam_end_b1] = grad_b1;
            adam_grad[adam_end_b1:adam_end_W2] = reshape(grad_W2, size(grad_W2));
            adam_grad[adam_end_W2:adam_end_b2] = grad_b2;
            adam_grad[adam_end_b2:adam_end_W3] = reshape(grad_W3, size(grad_W3));
            adam_grad[adam_end_W3:adam_end_b3] = grad_b3;

            beta1_pow *= beta1;
            beta2_pow *= beta2;

            adam_m = beta1 * adam_m + (1.0 - beta1) * adam_grad;
            adam_v = beta2 * adam_v + (1.0 - beta2) * (adam_grad * adam_grad);

            // Preserve the existing bias correction and epsilon convention.
            step_scale = lr * sqrt(1.0 - beta2_pow) / (1.0 - beta1_pow);
            eps_scaled = eps * sqrt(1.0 - beta2_pow);
            adam_params -= step_scale * adam_m / (sqrt(adam_v) + eps_scaled);

            // Restore layer tensors for the next forward and backward pass.
            W1 = reshape(adam_params[0:adam_end_W1], shape(W1)[0], shape(W1)[1]);
            b1 = adam_params[adam_end_W1:adam_end_b1];
            W2 = reshape(adam_params[adam_end_b1:adam_end_W2], shape(W2)[0], shape(W2)[1]);
            b2 = adam_params[adam_end_W2:adam_end_b2];
            W3 = reshape(adam_params[adam_end_b2:adam_end_W3], shape(W3)[0], shape(W3)[1]);
            b3 = adam_params[adam_end_W3:adam_end_b3];
            endSection(sec);

            endSection(sec_batch);
        }
    }

    // Consume losses from every step and all final parameters after training.
    print("Loss sum: ", declassify(loss_sum));
    print("Done.");

}

