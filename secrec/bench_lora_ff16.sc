import shared3p;
import shared3p_matrix;
import shared3p_random;
import stdlib;
import profiling;

import common;

domain pd_shared3p shared3p;


struct reluRes {
    pd_shared3p ff16[[2]] relu;
    pd_shared3p bool[[2]] mask;
}


reluRes reLU(pd_shared3p ff16[[2]] X) {
    reluRes res;

    pd_shared3p bool[[2]] m(shape(X)[0], shape(X)[1]);
    pd_shared3p ff16[[2]] r(shape(X)[0], shape(X)[1]);

    __syscall ("shared3p::relu_ff16_vec", __domainid (pd_shared3p), X, r, m);

    res.mask = m;
    res.relu = r;

    return res;
}


template <domain D : shared3p>
D ff16[[2]] mult (D ff16[[2]] x, D bool[[2]] y) {
    D ff16[[2]] res (shape(x)[0], shape(x)[1]);
    __syscall ("shared3p::choose_ff16_vec", __domainid (D), y, x, res, res);
    return res;
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


// uncomment to use BFP
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
    XY[:, :train_x.ncol] = train_x.data;
    XY[:, train_x.ncol:] = train_y.data;

    uint bs = 64;  // batch size
    pd_shared3p ff16[[2]] batch_x(bs, train_x.ncol);
    pd_shared3p ff16[[2]] batch_y(bs, train_y.ncol);

    uint input_dim = train_x.ncol;
    uint output_dim = train_y.ncol;
    uint epoch = 1;
    uint n_batch = 10;
    // uint n_batch = train_x.nrow / bs;
    pd_shared3p ff16 lr = 0.001;

    uint rank = 4;
    uint alpha = 4;
    pd_shared3p ff16 scale = (float32) rank / (float32) alpha;


    // Adam hyperparameters
    pd_shared3p ff16 beta1 = 0.9;
    pd_shared3p ff16 beta2 = 0.999;
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

    // LoRA
    pd_shared3p ff16[[2]] A = initWeights(rank, shape(W2)[1]) * 0.01;
    pd_shared3p ff16[[2]] B(shape(W2)[0], rank);
    pd_shared3p ff16[[2]] W2_LoRA = W2;

    // trainable
    pd_shared3p ff16[[2]] W3 = initWeights(output_dim, shape(W2)[0]);
    pd_shared3p ff16[[1]] b3 = initBias(output_dim);

    pd_shared3p ff16[[2]] zh1(bs, shape(W1)[0]);
    pd_shared3p ff16[[2]] zh2(bs, shape(W2)[0]);
    pd_shared3p bool[[2]] mask2(bs, shape(W2)[0]);
    pd_shared3p ff16[[2]] logitsProb(bs, output_dim);
    pd_shared3p ff16[[2]] delta(bs, output_dim);
    pd_shared3p ff16 loss;
    pd_shared3p ff16[[2]] grad_W3(output_dim, shape(W2)[0]);
    pd_shared3p ff16[[1]] grad_b3(output_dim);
    pd_shared3p ff16[[2]] grad_zh2(bs, shape(W2)[0]);
    pd_shared3p ff16[[2]] grad_W2(shape(W2)[0], shape(W1)[0]);
    pd_shared3p ff16[[2]] grad_A = A;
    pd_shared3p ff16[[2]] grad_B = B;

    // Adam state
    pd_shared3p ff16[[2]] m_A(shape(A)[0], shape(A)[1]);
    pd_shared3p ff16[[2]] v_A(shape(A)[0], shape(A)[1]);
    pd_shared3p ff16[[2]] m_B(shape(B)[0], shape(B)[1]);
    pd_shared3p ff16[[2]] v_B(shape(B)[0], shape(B)[1]);
    pd_shared3p ff16[[2]] m_W3(shape(W3)[0], shape(W3)[1]);
    pd_shared3p ff16[[2]] v_W3(shape(W3)[0], shape(W3)[1]);
    pd_shared3p ff16[[1]] m_b3(size(b3));
    pd_shared3p ff16[[1]] v_b3(size(b3));
    pd_shared3p ff16 step_scale;



    for (uint i=0; i< epoch; i++) {
        print("epoch: ", i+1, "/", epoch);

        for (uint j=0; j<n_batch; j++) {
            print("- batch: ", j+1, "/", n_batch);

            uint32 sec_batch = section("batch");

            batch_x = XY[j * bs: (j+1) * bs, :input_dim];
            batch_y = XY[j * bs: (j+1) * bs, input_dim:];

            // forward
            uint32 sec = section("forward");
            zh1 = addBias(matMult(batch_x, transpose(W1)), b1);
            reluRes r = reLU(zh1);

            zh1 = r.relu;

            W2_LoRA = W2 + scale * matMult(B, A);

            zh2 = addBias(matMult(zh1, transpose(W2_LoRA)), b2);

            r = reLU(zh2);
            mask2 = r.mask;
            zh2 = r.relu;

            logitsProb = addBias(matMult(zh2, transpose(W3)), b3);

            endSection(sec);
            sec = section("loss");

            logitsProb = softmax(logitsProb);
            loss = crossEntropyError(batch_y, logitsProb);
            delta = crossEntropyDelta(batch_y, logitsProb);
            print(" Loss: ", declassify(loss));
            endSection(sec);

            // backward
            sec = section("backward");
            grad_W3 = matMult(transpose(delta), zh2);
            grad_b3 = colSums(delta);

            grad_zh2 = matMult(delta, W3);
            grad_zh2 = mult(grad_zh2, mask2);
            grad_W2 = matMult(transpose(grad_zh2), zh1);
            grad_B = scale * matMult(grad_W2, transpose(A));
            grad_A = scale  * matMult(transpose(B), grad_W2);
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

    }

    print("Done.");

}
