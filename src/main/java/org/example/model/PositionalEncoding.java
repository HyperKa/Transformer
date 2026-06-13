package org.example.model;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.io.Serializable;

public class PositionalEncoding implements Serializable {
    private INDArray peMatrix;

    public PositionalEncoding(int maxLen, int dim) {
        peMatrix = Nd4j.zeros(maxLen, dim);
        for (int pos = 0; pos < maxLen; pos++) {
            for (int i = 0; i < dim / 2; i++) {
                double div = Math.pow(10000, (2.0 * i) / dim);
                peMatrix.putScalar(new int[]{pos, 2 * i}, Math.sin(pos / div));
                peMatrix.putScalar(new int[]{pos, 2 * i + 1}, Math.cos(pos / div));
            }
        }
    }

    public INDArray addPositionalEncoding(INDArray input) {
        int len = (int) input.size(0);
        // Извлекаем нужный фрагмент и прибавляем к входу
        return input.add(peMatrix.get(org.nd4j.linalg.indexing.NDArrayIndex.interval(0, len)));
    }
}