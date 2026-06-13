package org.example.model;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.example.repo.ParameterContainer; // Убедитесь, что путь корректен
import java.io.Serializable;
import java.util.*;

public class SelfAttention implements ParameterContainer, Serializable {
    private int embeddingDim;
    private int headDim;

    private INDArray Wq, Wk, Wv;
    private INDArray dWq, dWk, dWv;

    public SelfAttention(int embeddingDim, int headDim) {
        this.embeddingDim = embeddingDim;
        this.headDim = headDim;

        // Xavier инициализация: лимит = sqrt(6 / (in + out))
        double limit = Math.sqrt(6.0 / (embeddingDim + headDim));

        // ND4J: rand(rows, cols) создает значения от 0 до 1.
        // Приводим к диапазону [-limit, limit]: (rand * 2 * limit) - limit
        this.Wq = Nd4j.rand(embeddingDim, headDim).muli(2 * limit).subi(limit);
        this.Wk = Nd4j.rand(embeddingDim, headDim).muli(2 * limit).subi(limit);
        this.Wv = Nd4j.rand(embeddingDim, headDim).muli(2 * limit).subi(limit);

        this.dWq = Nd4j.zeros(embeddingDim, headDim);
        this.dWk = Nd4j.zeros(embeddingDim, headDim);
        this.dWv = Nd4j.zeros(embeddingDim, headDim);
    }

    public ForwardResult forward(INDArray input, int[] attentionMask) {
        // Q = X * Wq, K = X * Wk, V = X * Wv
        INDArray Q = input.mmul(Wq);
        INDArray K = input.mmul(Wk);
        INDArray V = input.mmul(Wv);

        // Scores = (Q * K^T) / sqrt(headDim)
        INDArray scores = Q.mmul(K.transpose()).divi(Math.sqrt(headDim));

        // Маскирование PAD токенов
        if (attentionMask != null) {
            for (int j = 0; j < attentionMask.length; j++) {
                if (attentionMask[j] == 0) {
                    scores.putScalar(new int[]{0, j}, -1e9); // Упрощенно для примера
                    // В реальном батче нужно использовать тензорную маску
                }
            }
        }

        INDArray attentionWeights = Transforms.softmax(scores, true);
        INDArray output = attentionWeights.mmul(V);

        Map<String, Object> cache = new HashMap<>();
        cache.put("input", input);
        cache.put("K", K);
        cache.put("Q", Q);
        cache.put("V", V);
        cache.put("attention_weights", attentionWeights);

        return new ForwardResult(output, cache);
    }

    public INDArray backward(INDArray grad_output, Map<String, Object> cache) {
        INDArray input = (INDArray) cache.get("input");
        INDArray K = (INDArray) cache.get("K");
        INDArray Q = (INDArray) cache.get("Q");
        INDArray V = (INDArray) cache.get("V");
        INDArray weights = (INDArray) cache.get("attention_weights");

        // dV = weights^T * grad_output
        INDArray dV = weights.transpose().mmul(grad_output);

        // Градиент Softmax (упрощенная векторная форма)
        INDArray dAttentionWeights = grad_output.mmul(V.transpose());
        INDArray dScores = weights.mul(dAttentionWeights.sub(weights.mul(dAttentionWeights).sum(1).reshape(weights.rows(), 1)));
        dScores.divi(Math.sqrt(headDim));

        INDArray dQ = dScores.mmul(K);
        INDArray dK = dScores.transpose().mmul(Q);

        // Обновление градиентов весов (dW = X^T * dY)
        this.dWq.addi(input.transpose().mmul(dQ));
        this.dWk.addi(input.transpose().mmul(dK));
        this.dWv.addi(input.transpose().mmul(dV));

        // Градиент по входу (dX = dY * W^T)
        return dQ.mmul(Wq.transpose())
                .addi(dK.mmul(Wk.transpose()))
                .addi(grad_output.mmul(Wv.transpose()));
    }

    @Override public List<INDArray> getParameters() { return List.of(Wk, Wq, Wv); }
    @Override public List<INDArray> getGradients() { return List.of(dWk, dWq, dWv); }
    @Override public void zeroGradients() { dWk.assign(0); dWq.assign(0); dWv.assign(0); }
}