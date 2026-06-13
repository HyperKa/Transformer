package org.example.model;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.example.repo.ParameterContainer;
import java.io.Serializable;
import java.util.*;

public class FeedForwardNetwork implements ParameterContainer, Serializable {
    private INDArray W1, b1, W2, b2;
    private INDArray dW1, db1, dW2, db2;

    public FeedForwardNetwork(int embeddingDim, int hiddenDim) {
        // Xavier/Glorot инициализация: limit = sqrt(6 / (in + out))
        double limit1 = Math.sqrt(6.0 / (embeddingDim + hiddenDim));
        this.W1 = Nd4j.rand(embeddingDim, hiddenDim).muli(2 * limit1).subi(limit1);
        this.b1 = Nd4j.zeros(1, hiddenDim);

        double limit2 = Math.sqrt(6.0 / (hiddenDim + embeddingDim));
        this.W2 = Nd4j.rand(hiddenDim, embeddingDim).muli(2 * limit2).subi(limit2);
        this.b2 = Nd4j.zeros(1, embeddingDim);

        this.dW1 = Nd4j.zeros(embeddingDim, hiddenDim);
        this.db1 = Nd4j.zeros(1, hiddenDim);
        this.dW2 = Nd4j.zeros(hiddenDim, embeddingDim);
        this.db2 = Nd4j.zeros(1, embeddingDim);
    }

    public ForwardResult forward(INDArray input) {
        // Слой 1: Linear + ReLU
        INDArray h = input.mmul(W1).addiRowVector(b1);
        INDArray h_relu = Transforms.relu(h, true); // true = inplace

        // Слой 2: Linear
        INDArray output = h_relu.mmul(W2).addiRowVector(b2);

        Map<String, Object> cache = new HashMap<>();
        cache.put("input", input);
        cache.put("h_relu", h_relu);
        return new ForwardResult(output, cache);
    }

    public INDArray backward(INDArray grad_output, Map<String, Object> cache) {
        INDArray input = (INDArray) cache.get("input");
        INDArray h_relu = (INDArray) cache.get("h_relu");

        // dW2 = h_relu^T * grad_output
        this.dW2.addi(h_relu.transpose().mmul(grad_output));
        this.db2.addi(grad_output.sum(0));
        INDArray dh_relu = grad_output.mmul(W2.transpose());

        // Backward ReLU: dh = dh_relu * (h_relu > 0)
        INDArray dh = dh_relu.mul(h_relu.gt(0));

        this.dW1.addi(input.transpose().mmul(dh));
        this.db1.addi(dh.sum(0));

        return dh.mmul(W1.transpose());
    }

    @Override public List<INDArray> getParameters() { return List.of(W1, W2, b1, b2); }
    @Override public List<INDArray> getGradients() { return List.of(dW1, dW2, db1, db2); }
    @Override public void zeroGradients() { dW1.assign(0); dW2.assign(0); db1.assign(0); db2.assign(0); }
}