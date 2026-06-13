package org.example.model;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class CrossEntropyLoss {
    public double forward(INDArray logits, int yTrue) {
        // Softmax + Log
        INDArray probs = Transforms.softmax(logits);
        return -Math.log(probs.getDouble(0, yTrue) + 1e-10);
    }

    public INDArray backward(INDArray logits, int yTrue) {
        // Градиент CrossEntropy + Softmax = (P - Y)
        INDArray probs = Transforms.softmax(logits);
        double oldVal = probs.getDouble(0, yTrue);
        probs.putScalar(0, yTrue, oldVal - 1.0);
        return probs; // dL/dLogits
    }
}