import shared3p;
import shared3p_string;
import shared3p_matrix;
import shared3p_random;
import shared3p_permutation;
import stdlib;
import profiling;


uint32 section(string name) {
    uint32 st = newSectionType(name);
    uint32 sec = startSection(st, 1::uint);
    return sec;
}


template<domain D : shared3p>
struct ResultMat {
    D float32[[2]] mat;   // even if it's a vector, saved as a matrix.
    Status status;
}


template<domain D : shared3p, type T>
D T[[2]] addBias(D T[[2]] X, D T[[1]] b) {
    uint bs = shape(X)[0];
    return reshape(reshape(X, size(X)) + _copyBlock(b, bs), bs, shape(X)[1]);
}

template <domain D : shared3p>
D float32[[2]] reLU(D float32[[2]] X) {
    __syscall("shared3p::relu_float32_vec", __domainid(D), X, X);
    return X;
}

template <domain D : shared3p>
D fix32[[2]] reLU(D fix32[[2]] X) {
    D bool[[2]] mask = reLUMask(X);
    __syscall("shared3p::relu_fix32_vec", __domainid(D), X, mask, X);
    return X;
}

template <domain D : shared3p>
D fix64[[2]] reLU(D fix64[[2]] X) {
    D bool[[2]] mask = reLUMask(X);
    __syscall("shared3p::relu_fix64_vec", __domainid(D), X, mask, X);
    return X;
}

template <domain D : shared3p>
D bool[[2]] reLUMask(D float32[[2]] X) {
    D bool[[2]] out(shape(X)[0], shape(X)[1]);
    __syscall("shared3p::relu_mask_float32_vec", __domainid(D), X, out);
    return out;
}

template <domain D : shared3p>
D bool[[2]] reLUMask(D fix32[[2]] X) {
    D bool[[2]] out(shape(X)[0], shape(X)[1]);
    __syscall("shared3p::relu_mask_fix32_vec", __domainid(D), X, out);
    return out;
}

template <domain D : shared3p>
D bool[[2]] reLUMask(D fix64[[2]] X) {
    D bool[[2]] out(shape(X)[0], shape(X)[1]);
    __syscall("shared3p::relu_mask_fix64_vec", __domainid(D), X, out);
    return out;
}

template<domain D : shared3p, type T>
D T[[2]] _normalize(D T[[2]] X) {
    // subtract the max value
    D T _max = max(reshape(X, size(X)));
    return X - _max;
}

template<domain D : shared3p, type T>
D T[[2]] softmax(D T[[2]] Z) {
    Z = _normalize(Z); // to prevent overflow
    D T[[2]] expZ = exp(Z);
    D T[[1]] rSums = rowSums(expZ);

    return expZ * _getVTiled(1 / rSums, shape(Z)[1]);
}

template<domain D : shared3p, type T>
D T crossEntropyError(D T[[2]] Y, D T[[2]] prob) {
    uint bs = shape(Y)[0];
    D T loss = - sum(rowSums(Y * ln(prob + 1e-7)));
    D float32 _bs = 1 / (float32) bs;
    D T __bs = (T) _bs;

    return loss * __bs;
}

template<domain D : shared3p, type T>
D T[[2]] crossEntropyDelta(D T[[2]] Y, D T[[2]] prob) {
    uint bs = shape(Y)[0];
    D float32 _bs = 1 / (float32) bs;
    D T __bs = (T) _bs;

    return (prob - Y) * __bs;
}

template<domain D : shared3p, type T>
D T[[2]] initWeights(uint nrow, uint ncol) {
    D int16[[1]] vec(nrow * ncol);
    int16[[1]] rand = declassify(randomize(vec));
    int16 std = 100;
    rand = rand % std - (std / 2) + 1;

    D int16[[1]] temp = rand;

    return reshape((T)temp * 0.0001, nrow, ncol);    
}

template<domain D : shared3p, type T>
D T[[1]] initBias(uint s) {
    D T[[1]] bias(s);
    return bias;
}

template<domain D : shared3p, type T>
struct Data {
    D T[[2]] data;
    uint nrow;
    uint ncol;
}

template<domain D : shared3p, type T>
Data<D, T> readArgData(string name, uint rows, uint cols) {
    D T[[1]] in = argument(name);
    Data<D, T> result;
    result.data = reshape(in, rows, cols);
    result.nrow = rows;
    result.ncol = cols;

    return result;
}


