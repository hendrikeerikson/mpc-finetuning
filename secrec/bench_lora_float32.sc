import shared3p;
import shared3p_matrix;
import stdlib;
import profiling;

import common;

domain pd_shared3p shared3p;


void main() {
    pd_shared3p float32 proxy;
    Data<pd_shared3p, float32> train_x = readArgData("x_train", 11800::uint, 784::uint);
    Data<pd_shared3p, float32> train_y = readArgData("y_train", 11800::uint, 2::uint);
    Data<pd_shared3p, float32> test_x = readArgData("x_test", 100::uint, 784::uint);
    Data<pd_shared3p, float32> test_y = readArgData("y_test", 100::uint, 2::uint);
    pd_shared3p float32[[2]] XY(train_x.nrow, train_x.ncol + train_y.ncol);
    XY[:, :train_x.ncol] = train_x.data;
    XY[:, train_x.ncol:] = train_y.data;

    uint bs = 64;  // batch size
    pd_shared3p float32[[2]] batch_x(bs, train_x.ncol);
    pd_shared3p float32[[2]] batch_y(bs, train_y.ncol);

    uint input_dim = train_x.ncol;
    uint output_dim = train_y.ncol;
    uint epoch = 1;
    uint n_batch = 10;
    // uint n_batch = train_x.nrow / bs;
    float32 lr = 0.001;

    uint rank = 4;
    uint alpha = 4;
    float32 scale = (float32) rank / (float32) alpha;

    // Adam hyperparameters
    pd_shared3p float32 beta1 = 0.9;
    pd_shared3p float32 beta2 = 0.999;
    pd_shared3p float32 eps = 0.0001;
    pd_shared3p float32 beta1_pow = 1.0;
    pd_shared3p float32 beta2_pow = 1.0;

    // pre-trained weights (fixed)
    pd_shared3p float32[[1]] W1_flat = argument("fc1_W");
    pd_shared3p float32[[2]] W1 = reshape(W1_flat, 32, 784);
    pd_shared3p float32[[1]] b1 = argument("fc1_b");

    pd_shared3p float32[[1]] W2_flat = argument("fc2_W");
    pd_shared3p float32[[2]] W2 = reshape(W2_flat, 16, 32);
    pd_shared3p float32[[1]] b2 = argument("fc2_b");

    // LoRA
    pd_shared3p float32[[2]] A = initWeights(rank, shape(W2)[1]) * 0.01;
    pd_shared3p float32[[2]] B(shape(W2)[0], rank);
    pd_shared3p float32[[2]] W2_LoRA = W2;

    // trainable
    pd_shared3p float32[[2]] W3 = initWeights(output_dim, shape(W2)[0]);
    pd_shared3p float32[[1]] b3 = initBias(output_dim);

    pd_shared3p float32[[2]] zh1(bs, shape(W1)[0]);
    pd_shared3p float32[[2]] zh2(bs, shape(W2)[0]);
    pd_shared3p bool[[2]] mask2(bs, shape(W2)[0]);
    pd_shared3p float32[[2]] logitsProb(bs, output_dim);
    pd_shared3p float32[[2]] delta(bs, output_dim);
    pd_shared3p float32 loss;
    pd_shared3p float32[[2]] grad_W3(output_dim, shape(W2)[0]);
    pd_shared3p float32[[1]] grad_b3(output_dim);
    pd_shared3p float32[[2]] grad_zh2(bs, shape(W2)[0]);
    pd_shared3p float32[[2]] grad_W2(shape(W2)[0], shape(W1)[0]);
    pd_shared3p float32[[2]] grad_A = A;
    pd_shared3p float32[[2]] grad_B = B;

    // Adam state
    pd_shared3p float32[[2]] m_A(shape(A)[0], shape(A)[1]);
    pd_shared3p float32[[2]] v_A(shape(A)[0], shape(A)[1]);
    pd_shared3p float32[[2]] m_B(shape(B)[0], shape(B)[1]);
    pd_shared3p float32[[2]] v_B(shape(B)[0], shape(B)[1]);
    pd_shared3p float32[[2]] m_W3(shape(W3)[0], shape(W3)[1]);
    pd_shared3p float32[[2]] v_W3(shape(W3)[0], shape(W3)[1]);
    pd_shared3p float32[[1]] m_b3(size(b3));
    pd_shared3p float32[[1]] v_b3(size(b3));
    pd_shared3p float32 step_scale;

    // for accuracy check
    pd_shared3p float32[[2]] _zh1(test_x.nrow, shape(W1)[0]);
    pd_shared3p float32[[2]] _zh2(test_x.nrow, shape(W2)[0]);
    pd_shared3p float32[[2]] pred(test_x.nrow, output_dim);
    pd_shared3p bool[[1]] correct(test_x.nrow);
    pd_shared3p float32 acc;



    for (uint i=0; i< epoch; i++) {
        print("epoch: ", i+1, "/", epoch);

        for (uint j=0; j<n_batch; j++) {
            print("- batch: ", j+1, "/", n_batch);

            uint32 sec_batch = section("batch");
            batch_x = XY[j * bs: (j+1) * bs, :input_dim];
            batch_y = XY[j * bs: (j+1) * bs, input_dim:];

            // forward
            uint32 sec = section("forward");
            zh1 = addBias(matrixMultiplication(batch_x, transpose(W1)), b1);
            zh1 = reLU(zh1);

            W2_LoRA = W2 + scale * matrixMultiplication(B, A);

            zh2 = addBias(matrixMultiplication(zh1, transpose(W2_LoRA)), b2);
            mask2 = reLUMask(zh2);
            zh2 = reLU(zh2);

            logitsProb = addBias(matrixMultiplication(zh2, transpose(W3)), b3);

            endSection(sec);
            sec = section("loss");

            logitsProb = softmax(logitsProb);
            loss = crossEntropyError(batch_y, logitsProb);
            delta = crossEntropyDelta(batch_y, logitsProb);
            print(" Loss: ", declassify(loss));
            endSection(sec);

            // backward
            sec = section("backward");
            grad_W3 = matrixMultiplication(transpose(delta), zh2);
            grad_b3 = colSums(delta);

            grad_zh2 = matrixMultiplication(delta, W3);
            grad_zh2 *= (float32) mask2;
            grad_W2 = matrixMultiplication(transpose(grad_zh2), zh1);
            grad_B = scale * matrixMultiplication(grad_W2, transpose(A));
            grad_A = scale  * matrixMultiplication(transpose(B), grad_W2);
            endSection(sec);


            // Adam
            sec = section("adam");
            beta1_pow *= beta1;
            beta2_pow *= beta2;

            m_A = beta1 * m_A + (1.0 - beta1) * grad_A;
            v_A = beta2 * v_A + (1.0 - beta2) * (grad_A * grad_A);
            m_B = beta1 * m_B + (1.0 - beta1) * grad_B;
            v_B = beta2 * v_B + (1.0 - beta2) * (grad_B * grad_B);
            m_W3 = beta1 * m_W3 + (1.0 - beta1) * grad_W3;
            v_W3 = beta2 * v_W3 + (1.0 - beta2) * (grad_W3 * grad_W3);
            m_b3 = beta1 * m_b3 + (1.0 - beta1) * grad_b3;
            v_b3 = beta2 * v_b3 + (1.0 - beta2) * (grad_b3 * grad_b3);

            // Bias-corrected Adam step.
            step_scale = lr * sqrt(1.0 - beta2_pow) / (1.0 - beta1_pow);
            A  -= step_scale * m_A / (sqrt(v_A) + eps);
            B  -= step_scale * m_B / (sqrt(v_B) + eps);
            W3 -= step_scale * m_W3 / (sqrt(v_W3) + eps);
            b3 -= step_scale * m_b3 / (sqrt(v_b3) + eps);
            endSection(sec);

            endSection(sec_batch);
        }

        // check accuracy
        _zh1 = addBias(matrixMultiplication(test_x.data, transpose(W1)), b1);
        _zh1 = reLU(_zh1);
        W2_LoRA = W2 + scale * matrixMultiplication(B, A);
        _zh2 = addBias(matrixMultiplication(_zh1, transpose(W2_LoRA)), b2);
        _zh2 = reLU(_zh2);
        pred = addBias(matrixMultiplication(_zh2, transpose(W3)), b3);
        correct = pred[:, 0] > pred[:, 1];
        correct = (float32)correct == test_y.data[:, 0];
        acc = sum((float32)correct) / (float32)test_x.nrow;
        pred = softmax(pred);
        loss = crossEntropyError(test_y.data, pred);
        print("Test accuracy: ", declassify(acc));
        print("Test loss: ", declassify(loss));

    }

    print("Done.");

}

