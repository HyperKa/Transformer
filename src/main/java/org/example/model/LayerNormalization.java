package org.example.model;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.example.repo.ParameterContainer;

import java.io.Serializable;
import java.util.*;

public class LayerNormalization implements ParameterContainer, Serializable {
    private INDArray gamma, beta;
    private INDArray dGamma, dBeta;
    private double epsilon;

    public LayerNormalization(int dim, double epsilon) {
        this.gamma = Nd4j.ones(1, dim);
        this.beta = Nd4j.zeros(1, dim);
        this.dGamma = Nd4j.zeros(1, dim);
        this.dBeta = Nd4j.zeros(1, dim);
        this.epsilon = epsilon;
    }

    public ForwardResult forward(INDArray input) {
        INDArray mean = input.mean(1).reshape(input.rows(), 1);
        INDArray var = input.var(false, 1).reshape(input.rows(), 1);
        INDArray stdInv = Transforms.pow(var.add(epsilon), -0.5);

        INDArray xHat = input.subColumnVector(mean).muliColumnVector(stdInv);
        INDArray output = xHat.mulRowVector(gamma).addiRowVector(beta);

        Map<String, Object> cache = new HashMap<>();
        cache.put("xHat", xHat);
        cache.put("stdInv", stdInv);
        cache.put("input", input);
        cache.put("mean", mean);
        return new ForwardResult(output, cache);
    }

    public INDArray backward(INDArray gradOutput, Map<String, Object> cache) {
        INDArray xHat = (INDArray) cache.get("xHat");
        INDArray stdInv = (INDArray) cache.get("stdInv");
        INDArray input = (INDArray) cache.get("input");
        INDArray mean = (INDArray) cache.get("mean");
        long N = input.columns(); // embeddingDim (64)

        // 1. Градиенты параметров (суммируем по строкам)
        this.dGamma.addi(gradOutput.mul(xHat).sum(0));
        this.dBeta.addi(gradOutput.sum(0));

        // 2. Градиент по xHat
        INDArray dxHat = gradOutput.mulRowVector(gamma);

        // 3. Градиент по дисперсии (используем mul вместо muli, чтобы не сломать форму)
        INDArray dvar = dxHat.mul(input.subColumnVector(mean))
                .sum(1).reshape(input.rows(), 1)
                .mul(-0.5)
                .mul(Transforms.pow(stdInv, 3));

        // 4. Градиент по среднему
        INDArray dmean = dxHat.mulColumnVector(stdInv).mul(-1).sum(1).reshape(input.rows(), 1)
                .add(dvar.mul(input.subColumnVector(mean).mul(-2).sum(1).reshape(input.rows(), 1)).div(N));

        // 5. Финальный градиент по входу (dx)
        // Здесь важно использовать обычный mul, так как мы умножаем [150, 1] на [150, 64]
        INDArray dx = dxHat.mulColumnVector(stdInv)
                .add(dvar.mul(2).mul(input.subColumnVector(mean)).div(N))
                .add(dmean.div(N));

        return dx;
    }

    public void setParams(INDArray gamma, INDArray beta) {
        this.gamma = gamma;
        this.beta = beta;
    }

    @Override public List<INDArray> getParameters() { return List.of(gamma, beta); }
    @Override public List<INDArray> getGradients() { return List.of(dGamma, dBeta); }
    @Override public void zeroGradients() { dGamma.assign(0); dBeta.assign(0); }
}