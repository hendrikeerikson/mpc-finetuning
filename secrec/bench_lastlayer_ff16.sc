import shared3p;
import shared3p_matrix;
import shared3p_random;
import stdlib;
import profiling;

import common;

domain pd_shared3p shared3p;



template <domain D : shared3p>
D ff16[[2]] reLU(D ff16[[2]] X) {
    pd_shared3p bool[[1]] mask(size(X));
    __syscall ("shared3p::relu_ff16_vec", __domainid (D), X, X, mask);

    return X;
}


template<domain D : shared3p>
D ff16 crossEntropyError(D ff16[[2]] Y, D ff16[[2]] prob) {
    uint bs = shape(Y)[0];
    D ff16 loss = - sum(rowSums(Y * ln(prob + 1e-7)));
    float32 _bs = 1 / (float32) bs;
    D ff16 __bs = _bs;

    return loss * __bs;
}


template<domain D : shared3p>
D ff16[[2]] crossEntropyDelta(D ff16[[2]] Y, D ff16[[2]] prob) {
    uint bs = shape(Y)[0];
    float32 _bs = 1 / (float32) bs;
    D ff16 __bs = _bs;

    return (prob - Y) * __bs;
}


template<domain D : shared3p>
D ff16[[2]] initWeights(uint nrow, uint ncol) {
    D int16[[1]] vec(nrow * ncol);
    int16[[1]] rand = declassify(randomize(vec));
    int16 std = 100;
    rand = rand % std - (std / 2) + 1;

    float32[[1]] temp = ((float32)rand) * 0.0001;
    D ff16[[1]] res = temp;

    return reshape(res, nrow, ncol);
}


template <domain D : shared3p>
D ff16[[2]] BFPMatMult (D ff16[[2]] x, D ff16[[2]] y) {
    assert (shape(x)[1] == shape(y)[0]);
    uint M = shape(x)[0];
    uint N = shape(x)[1];
    uint K = shape(y)[1];

    pd_shared3p ff16[[2]] res(M, K);
    __syscall("shared3p::bfp_mat_mult_ff16", __domainid(pd_shared3p), x, y, res, __cref M, __cref N, __cref K);

    return res;
}


template<domain D : shared3p>
D ff16[[2]] softmax(D ff16[[2]] Z) {
    D ff16[[2]] expZ = exp(Z);
    D ff16[[1]] rSums = rowSums(expZ);

    return expZ * _getVTiled(1 / rSums, shape(Z)[1]);
}


// Uncomment to use BFP
template <domain D : shared3p>
D ff16[[2]] matMult(D ff16[[2]] x, D ff16[[2]] y) {
    // return BFPMatMult(x, y);
    return matrixMultiplication(x, y);
}


void main() {

    Data<pd_shared3p, ff16> train_x = readArgData("x_train", 11800::uint, 784::uint);
    Data<pd_shared3p, ff16> train_y = readArgData("y_train", 11800::uint, 2::uint);
    Data<pd_shared3p, ff16> test_x = readArgData("x_test", 100::uint, 784::uint);
    Data<pd_shared3p, ff16> test_y = readArgData("y_test", 100::uint, 2::uint);
    pd_shared3p ff16[[2]] XY(train_x.nrow, train_x.ncol + train_y.ncol);


    uint bs = 64;  // batch size
    pd_shared3p ff16[[2]] batch_x(bs, train_x.ncol);
    pd_shared3p ff16[[2]] batch_y(bs, train_y.ncol);

    uint input_dim = train_x.ncol;
    uint output_dim = train_y.ncol;
    uint epoch = 1;
    uint n_batch = train_x.nrow / bs;
    // uint n_batch = 10;
    pd_shared3p ff16 lr = 0.001;


    // Adam hyperparameters
    pd_shared3p ff16 beta1 = 0.9;
    pd_shared3p ff16 beta2 = 0.999;
    pd_shared3p ff16 inv_beta1 = 1 - beta1;
    pd_shared3p ff16 inv_beta2 = 1 - beta2;
    pd_shared3p ff16 eps = 0.0001;
    pd_shared3p ff16 beta1_pow = 1.0;
    pd_shared3p ff16 beta2_pow = 1.0;

    // pre-trained weights (fixed)
    pd_shared3p ff16[[1]] W1_flat = argument("fc1_W");
    pd_shared3p ff16[[2]] W1 = reshape(W1_flat, 32, 784);
    pd_shared3p ff16[[1]] b1 = argument("fc1_b");

    pd_shared3p ff16[[1]] W2_flat = argument("fc2_W");
    pd_shared3p ff16[[2]] W2 = reshape(W2_flat, 16, 32);
    pd_shared3p ff16[[1]] b2 = argument("fc2_b");

    // trainable
    pd_shared3p ff16[[2]] W3 = initWeights(output_dim, shape(W2)[0]);
    pd_shared3p ff16[[1]] b3 = initBias(output_dim);


    pd_shared3p ff16[[2]] zh1(bs, shape(W1)[0]);
    pd_shared3p ff16[[2]] zh2(bs, shape(W2)[0]);
    pd_shared3p ff16[[2]] logitsProb(bs, output_dim);
    pd_shared3p ff16[[2]] delta(bs, output_dim);
    pd_shared3p ff16 loss;
    pd_shared3p ff16[[2]] grad_W3(output_dim, shape(W2)[0]);
    pd_shared3p ff16[[1]] grad_b3(output_dim);

    // Adam state
    pd_shared3p ff16[[2]] m_W3(shape(W3)[0], shape(W3)[1]);
    pd_shared3p ff16[[2]] v_W3(shape(W3)[0], shape(W3)[1]);
    pd_shared3p ff16[[1]] m_b3(size(b3));
    pd_shared3p ff16[[1]] v_b3(size(b3));
    pd_shared3p ff16 step_scale;


    // for accuracy check
    pd_shared3p ff16[[2]] _zh1(test_x.nrow, shape(W1)[0]);
    pd_shared3p ff16[[2]] _zh2(test_x.nrow, shape(W2)[0]);
    pd_shared3p ff16[[2]] pred(test_x.nrow, output_dim);
    bool[[1]] correct(test_x.nrow);
    float32 acc;


    for (uint i=0; i< epoch; i++) {
        print("epoch: ", i+1, "/", epoch);
        // TODO: shuffle
        // XY = shuffleDatasets(train_x, train_y, i);

        XY[:, :train_x.ncol] = train_x.data;
        XY[:, train_x.ncol:] = train_y.data;

        for (uint j=0; j<n_batch; j++) {
            print("- batch: ", j+1, "/", n_batch);


            uint32 batch_sec = section("batch");
            batch_x = XY[j * bs: (j+1) * bs, :input_dim];
            batch_y = XY[j * bs: (j+1) * bs, input_dim:];

            // forward
            uint32 sec = section("forward");
            zh1 = addBias(matMult(batch_x, transpose(W1)), b1);
            zh1 = reLU(zh1);
            zh2 = addBias(matMult(zh1, transpose(W2)), b2);
            zh2 = reLU(zh2);

            logitsProb = addBias(matMult(zh2, transpose(W3)), b3);
            endSection(sec);

            sec = section("loss");

            logitsProb = softmax(logitsProb);
            loss = crossEntropyError(batch_y, logitsProb);
            delta = crossEntropyDelta(batch_y, logitsProb);
            print("   Loss: ", declassify(loss));
            endSection(sec);

            // backward
            sec = section("backward");
            grad_W3 = matMult(transpose(delta), zh2);
            grad_b3 = colSums(delta);
            endSection(sec);

            // Adam
            sec = section("adam");
            beta1_pow *= beta1;
            beta2_pow *= beta2;

            m_W3 = beta1 * m_W3 + (inv_beta1) * grad_W3;
            v_W3 = beta2 * v_W3 + (inv_beta2) * (grad_W3 * grad_W3);
            m_b3 = beta1 * m_b3 + (inv_beta1) * grad_b3;
            v_b3 = beta2 * v_b3 + (inv_beta2) * (grad_b3 * grad_b3);

            // Bias-corrected Adam step.
            step_scale = lr * sqrt(1.0 - beta2_pow) / (1.0 - beta1_pow);
            W3 -= step_scale * m_W3 / (sqrt(v_W3) + eps);
            b3 -= step_scale * m_b3 / (sqrt(v_b3) + eps);

            endSection(sec);
            endSection(batch_sec);
        }

        // check accuracy
        _zh1 = addBias(matMult(test_x.data, transpose(W1)), b1);
        _zh1 = reLU(_zh1);
        _zh2 = addBias(matMult(_zh1, transpose(W2)), b2);
        _zh2 = reLU(_zh2);
        pred = addBias(matMult(_zh2, transpose(W3)), b3);

        correct = declassify(pred[:, 0]) > declassify(pred[:, 1]);
        correct = (float32)correct == declassify(test_y.data[:, 0]);
        acc = sum((float32)correct) / (float32)test_x.nrow;
        pred = softmax(pred);
        loss = crossEntropyError(test_y.data, pred);
        print("Test accuracy: ", acc);
        print("Test loss: ", declassify(loss));
    }
    print("Done.");
}
