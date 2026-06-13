package org.example.model;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.example.repo.ParameterContainer;
import org.example.data.ConfigConstants;
import java.io.Serializable;
import java.util.*;

public class ClassificationHead implements ParameterContainer, Serializable {
    private INDArray W;  // Веса [inputDim, numClasses]
    private INDArray b;  // Смещение [1, numClasses]
    private INDArray dW;
    private INDArray db;

    public ClassificationHead(int inputDim, int numClasses) {
        double limit = Math.sqrt(6.0 / (inputDim + numClasses));
        // Xavier инициализация
        this.W = Nd4j.rand(new int[]{inputDim, numClasses}).muli(2 * limit).subi(limit);
        this.b = Nd4j.zeros(1, numClasses);
        this.dW = Nd4j.zeros(inputDim, numClasses);
        this.db = Nd4j.zeros(1, numClasses);
    }

    public ForwardResult forward(INDArray input) {
        // Y = X * W + b
        INDArray output = input.mmul(W).addiRowVector(b);

        Map<String, Object> cache = new HashMap<>();
        cache.put("input", input);

        return new ForwardResult(output, cache);
    }

    public INDArray backward(INDArray grad_output, Map<String, Object> cache) {
        INDArray input = (INDArray) cache.get("input");

        // dW = input^T * grad_output
        this.dW.addi(input.transpose().mmul(grad_output));
        // db = sum(grad_output) по строкам
        this.db.addi(grad_output.sum(0));

        // dX = grad_output * W^T
        return grad_output.mmul(W.transpose());
    }

    @Override public List<INDArray> getParameters() { return List.of(W, b); }
    @Override public List<INDArray> getGradients() { return List.of(dW, db); }
    @Override public void zeroGradients() { dW.assign(0); db.assign(0); }
}