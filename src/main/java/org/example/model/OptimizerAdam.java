package org.example.model;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.example.repo.ParameterContainer;
import java.util.ArrayList;
import java.util.List;

public class OptimizerAdam {
    private final ParameterContainer model;
    private double learningRate;
    private final double beta1 = 0.9;
    private final double beta2 = 0.999;
    private final double epsilon = 1e-8;
    private int t = 0;

    private List<INDArray> m;
    private List<INDArray> v;

    public OptimizerAdam(ParameterContainer model, double learningRate) {
        this.model = model;
        this.learningRate = learningRate;
        this.m = new ArrayList<>();
        this.v = new ArrayList<>();

        for (INDArray param : model.getParameters()) {
            m.add(Nd4j.zeros(param.shape()));
            v.add(Nd4j.zeros(param.shape()));
        }
    }

    public void step() {
        t++;
        List<INDArray> params = model.getParameters();
        List<INDArray> grads = model.getGradients();

        for (int i = 0; i < params.size(); i++) {
            INDArray theta = params.get(i);
            INDArray g = grads.get(i);

            // m = beta1 * m + (1 - beta1) * g
            m.get(i).muli(beta1).addi(g.mul(1 - beta1));
            // v = beta2 * v + (1 - beta2) * g^2
            v.get(i).muli(beta2).addi(Transforms.pow(g, 2).muli(1 - beta2));

            // Коррекция смещения
            INDArray mHat = m.get(i).div(1 - Math.pow(beta1, t));
            INDArray vHat = v.get(i).div(1 - Math.pow(beta2, t));

            // Обновление: theta = theta - lr * mHat / (sqrt(vHat) + eps)
            INDArray denom = Transforms.sqrt(vHat, false).addi(epsilon);
            theta.subi(mHat.div(denom).muli(learningRate));
        }
    }

    public void zeroGrad() {
        // Этот метод просто вызывает обнуление у всей модели
        model.zeroGradients();
    }
}