package org.example.model;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class CrossEntropyLoss {
    private Map<String, Object> cache;

    public double forward(RealMatrix logits, int y_true) {
        RealMatrix probabilities = applySoftmax(logits);  // получение вероятностей

        // вычисление функции потерь кросс энтропии - Loss = -log(вероятность правильного класса)
        double predicted_prob_for_true_class = probabilities.getEntry(0, y_true);
        double loss = -Math.log(predicted_prob_for_true_class + 1e-9);  // добавление малого значения, чтобы не вычислять логарифм от нуля

        this.cache = new HashMap<>();
        this.cache.put("probabilities", probabilities);
        this.cache.put("y_true", y_true);

        return loss;
    }

    public RealMatrix backward() {
        if (this.cache == null) {
            throw new IllegalStateException("CrossEntropyLoss.java: backward() обход нужно вызывать после forward(), cache пустой");
        }

        RealMatrix y_pred = (RealMatrix) this.cache.get("probabilities");
        int y_true = (int) this.cache.get("y_true");
        int numClasses = y_pred.getColumnDimension();

        RealMatrix grad = y_pred.copy();
        grad.addToEntry(0, y_true, -1.0);
        this.cache = null;

        return grad;
    }

    private RealMatrix applySoftmax(RealMatrix logits) {
        double[] logits_data = logits.getRow(0);
        double max_logit = Arrays.stream(logits_data).max().orElse(0.0);

        double[] exp_values = new double[logits_data.length];
        double sum_exp_values = 0.0;

        for(int i = 0; i < logits_data.length; i++) {
            exp_values[i] = Math.exp(logits_data[i] - max_logit); // Стабилизация
            sum_exp_values += exp_values[i];
        }

        double[] probabilities = new double[logits_data.length];
        for(int i=0; i < logits_data.length; i++) {
            probabilities[i] = exp_values[i] / sum_exp_values;
        }

        return new Array2DRowRealMatrix(new double[][]{probabilities});
    }
}
