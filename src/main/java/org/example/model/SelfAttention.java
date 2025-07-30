package org.example.model;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.StatUtils;
import org.example.ParameterContainer;
import org.example.data.ConfigConstants;

import java.util.*;

public class SelfAttention implements ParameterContainer {

    private int embeddingDim;
    private int headDim;

    private RealMatrix Wq, dWq;
    private RealMatrix Wk, dWk;
    private RealMatrix Wv, dWv;

    public SelfAttention(int embeddingDim, int headDim) {
        this.headDim = headDim;
        this.embeddingDim = embeddingDim;

        this.Wq = createRandomMatrix(embeddingDim, headDim);
        this.Wk = createRandomMatrix(embeddingDim, headDim);
        this.Wv = createRandomMatrix(embeddingDim, headDim);

        this.dWq = new Array2DRowRealMatrix(embeddingDim, headDim);  // градиенты инициализированы нулями
        this.dWk = new Array2DRowRealMatrix(embeddingDim, headDim);
        this.dWv = new Array2DRowRealMatrix(embeddingDim, headDim);
    }


    private RealMatrix createRandomMatrix0(int embeddingDim, int headDim) {
        double[] randomData = new Random().doubles((long) embeddingDim * headDim, -0.1, 0.1).toArray();
        double[][] data2D = new double[embeddingDim][headDim];
        //RealMatrix data2D_2 = new Array2DRowRealMatrix(embeddingDim, headDim);
        int k = 0;
        for (int i = 0; i < embeddingDim; i++) {
            for (int j = 0; j < headDim; j++) {
                data2D[i][j] = randomData[k++];
                //data2D_2.addToEntry(i, j, randomData[k++]);  // добавление к 0 (элементам матрицы) значения из вектора randomData
                //data2D_2.setEntry(i, j, randomData[k++]);    // изменение 0 на элемент вектора, это вернее, но оба способа возможны
            }
        }
        return new Array2DRowRealMatrix(data2D);
    }

    private RealMatrix createRandomMatrix(int rows, int columns) {
        Random rand;
        if (ConfigConstants.RANDOM_SEED != null) {
            rand = new Random(ConfigConstants.RANDOM_SEED);
        } else {
            rand = new Random();
        }

        if (ConfigConstants.IS_TEST) {
            double step = 0.05;

            double[][] data2D = new double[rows][columns];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    int randomInt = rand.nextInt(5) - 2; // диапазон становится: [-2, -1, 0, 1, 2]
                    data2D[i][j] = randomInt * step;
                }
            }
            return new Array2DRowRealMatrix(data2D);

        } else {
            // --- ОБЫЧНЫЙ РЕЖИМ: непрерывные случайные числа, НО ВОСПРОИЗВОДИМЫЕ ---
            long requiredDataLength = (long) rows * columns;

            double[] randomData = rand.doubles(requiredDataLength, -0.1, 0.1).toArray();

            double[][] data2D = new double[rows][columns];
            int k = 0;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    data2D[i][j] = randomData[k++];
                    //data2D_2.addToEntry(i, j, randomData[k++]);  // добавление к 0 (элементам матрицы) значения из вектора randomData
                    //data2D_2.setEntry(i, j, randomData[k++]);    // изменение 0 на элемент вектора, это вернее, но оба способа возможны
                }
            }
            return new Array2DRowRealMatrix(data2D);
        }
    }


    public ForwardResult forward(RealMatrix inputEmbeddings, int[] attentionMask) {
        RealMatrix Q = inputEmbeddings.multiply(this.Wq);
        RealMatrix K = inputEmbeddings.multiply(this.Wk);
        RealMatrix V = inputEmbeddings.multiply(this.Wv);

        RealMatrix scopes = Q.multiply(K.transpose()).scalarMultiply(1.0 / Math.sqrt(this.headDim));  // Q*K^T * 1/sqrt(8)  // 100 на 8 * 8 на 100

        for (int i = 0; i < scopes.getRowDimension(); i++) {
            for (int j = 0; j < scopes.getColumnDimension(); j++) {
                if (attentionMask[j] == 0) {                            // если токен - PAD, то присваиваем большое отриц значение
                    scopes.addToEntry(i, j, -1e9);
                }
            }
        }

        RealMatrix attentionWeights = applySoftmax(scopes);
        RealMatrix output = attentionWeights.multiply(V);  // 100 на 100 * 100 на 8, на выходе матрица 100 на 8

        Map<String, Object> cache = new HashMap<>();
        cache.put("input", inputEmbeddings);  // x
        cache.put("K", K);
        cache.put("Q", Q);
        cache.put("V", V);
        cache.put("attention_weights", attentionWeights);

        return new ForwardResult(output, cache) ;
    }

    public RealMatrix backward(RealMatrix grad_output, Map<String, Object> cache) {
        RealMatrix input = (RealMatrix) cache.get("input");
        RealMatrix K = (RealMatrix) cache.get("K");
        RealMatrix Q = (RealMatrix) cache.get("Q");
        RealMatrix V = (RealMatrix) cache.get("V");
        RealMatrix attentionWeights = (RealMatrix) cache.get("attention_weights");

        // dL/d(attentionWeights) = dL/dy * dy/d(attentionWeights) ::: dL/dy * W^T
        RealMatrix dAttentionWeights = grad_output.multiply(V.transpose());

        // dL/dV = dL/dy * dy/dV ::: x^T * dL/dy
        RealMatrix dV = attentionWeights.transpose().multiply(grad_output);

        RealMatrix dscopes = softmax_backward(dAttentionWeights, attentionWeights);  // вычислен градиент softmax

        RealMatrix d_scale = dscopes.scalarMultiply(1.0 / Math.sqrt(this.headDim));

        RealMatrix dQ = d_scale.multiply(K);
        RealMatrix dK = d_scale.transpose().multiply(Q);

        // Тут опять применение уравнения dL/dW = x^T * dL/dy
        this.dWq = input.transpose().multiply(dQ);
        this.dWk = input.transpose().multiply(dK);
        this.dWv = input.transpose().multiply(dV);

        // А теперь, есть 3 пути, как может измениться ввод: через изменение матриц Q, K, V.
        // Поэтому dL/dx рассчитывается относительно трех матриц-весов W и Q, K, V соответственно,
        // рассчитываю вход относительно трех этих пар матриц и суммирую для финального градиента

        // Применение уравнения dL/dx = dL/dy * W^T
        RealMatrix x_from_Q = dQ.multiply(this.Wq.transpose());
        RealMatrix x_from_K = dK.multiply(this.Wk.transpose());
        RealMatrix x_from_V = dV.multiply(this.Wv.transpose());

        return x_from_K.add(x_from_Q).add(x_from_V);
    }



    public RealMatrix applySoftmax(RealMatrix scopes) {
        double[][] resultData = new double[scopes.getRowDimension()][scopes.getColumnDimension()];

        for (int i = 0; i < scopes.getRowDimension(); i++) {
            double[] row = scopes.getRow(i);
            double maxVal = Arrays.stream(row).max().orElse(0.0);
            double[] expValues = Arrays.stream(row).map(val -> Math.exp(val - maxVal)).toArray();
            double sumExpValues = Arrays.stream(expValues).sum();

            for (int j = 0; j < scopes.getColumnDimension(); j++) {
                resultData[i][j] = expValues[j] / sumExpValues;
            }
        }
        return new Array2DRowRealMatrix(resultData);
    }

    private RealMatrix softmax_backward(RealMatrix dAttentionWeights, RealMatrix attentionWeights) {
        RealMatrix dscores = new Array2DRowRealMatrix(dAttentionWeights.getRowDimension(), dAttentionWeights.getColumnDimension());

        for (int i = 0; i < dAttentionWeights.getRowDimension(); i++) {
            double[] dweights_row = dAttentionWeights.getRow(i);
            double[] weights_row = attentionWeights.getRow(i);

            // теперь с фактом того, что dweights вычислен в MultiHeadAttention, это является grad_output_i, weights <=> yi - scopes-матрица, которая нужна
            // Шаг 2.1: Вычисляем dL/dy * y (поэлементно)
            double[] d_times_y = new double[weights_row.length];
            for (int j = 0; j < weights_row.length; j++) {
                d_times_y[j] = dweights_row[j] * weights_row[j];  // grad_output_i * yi
            }

            // Шаг 2.2: Вычисляем скаляр sum(dL/dy * y)
            double sum_d_times_y = StatUtils.sum(d_times_y);  // sum(grad_output_i * yi)

            // Шаг 2.3: Вычисляем градиент для каждого элемента входа
            for (int j = 0; j < weights_row.length; j++) {
                // Формула: dscores_j = y_j * (dweights_j - sum_d_times_y)
                dscores.setEntry(i, j, weights_row[j] * (dweights_row[j] - sum_d_times_y));  // это yi * [grad_output_j - sum (...)]
            }

            // Если мне не будет в ломы, залью на гит свой блокнот с вычислениями и выводами формул, штука интересная, но столько действий, я в ахере

        }
        return dscores;
    }

    @Override
    public List<RealMatrix> getParameters() {
        // ArrayList<RealMatrix> result = new ArrayList<>();
        // result.add(this.Wk);
        // result.add(this.Wq);
        // result.add(this.Wv);
        // return result;
        return List.of(this.Wk, this.Wq, this.Wv);
    }

    @Override
    public List<RealMatrix> getGradients() {
        return List.of(this.dWk, this.dWq, this.dWv);
    }

    @Override
    public void zeroGradients() {
        this.dWk.scalarMultiply(0.0);
        this.dWq.scalarMultiply(0.0);
        this.dWv.scalarMultiply(0.0);
    }
}
