package org.example.model;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.example.ParameterContainer;

import org.example.data.ConfigConstants;

import java.util.*;

public class FeedForwardNetwork implements ParameterContainer {
    private RealMatrix W1;  // Веса первого слоя
    private RealMatrix b1;  // смещение слоя 1
    private RealMatrix W2;  // Веса второго слоя
    private RealMatrix b2;  // смещение слоя 2

    private RealMatrix dW1;
    private RealMatrix db1;
    private RealMatrix dW2;
    private RealMatrix db2;

    private int embeddingDim;
    private int hiddenDim;

    public   FeedForwardNetwork(int embeddingDim, int hiddenDim) {
        this.W1 = createRandomMatrix(embeddingDim, hiddenDim);  // расширяющий слой, hiddenDim > embeddingDim
        this.b1 = new Array2DRowRealMatrix(1, hiddenDim);
        this.W2 = createRandomMatrix(hiddenDim, embeddingDim);  // сужающий слой
        this.b2 = new Array2DRowRealMatrix(1, embeddingDim);

        this.dW1 = new Array2DRowRealMatrix(embeddingDim, hiddenDim);  // расширяющий слой, hiddenDim > embeddingDim
        this.db1 = new Array2DRowRealMatrix(1, hiddenDim);
        this.dW2 = new Array2DRowRealMatrix(hiddenDim, embeddingDim);  // сужающий слой
        this.db2 = new Array2DRowRealMatrix(1, embeddingDim);

        this.embeddingDim = embeddingDim;
        this.hiddenDim = hiddenDim;
    }

    public ForwardResult forward(RealMatrix input) {  // Поступает матрица после нормализации 100x64

        RealMatrix matrix_w1 = input.multiply(this.W1);
        for (int i = 0; i < matrix_w1.getRowDimension(); i++) {
            matrix_w1.setRowMatrix(i, matrix_w1.getRowMatrix(i).add(this.b1));
            // matrix_w1.getRowMatrix(i).add(this.b1);
        }

        // ReLU:
        RealMatrix matrix_w1_copy = matrix_w1.copy();
        for (int i = 0; i < matrix_w1.getRowDimension(); i++) {
            for (int j = 0; j < matrix_w1.getColumnDimension(); j++) {
                if (matrix_w1.getEntry(i, j) < 0.0) {
                    matrix_w1.setEntry(i, j, 0.0);
                }
            }
        }

        RealMatrix output = matrix_w1.multiply(this.W2);  //
        for (int i = 0; i < output.getRowDimension(); i++) {
            output.setRowMatrix(i, output.getRowMatrix(i).add(this.b2));  // get возвращает матрицу-строку
            // matrix_w1.getRowMatrix(i).add(this.b2);
        }

        Map<String, Object> cache = new HashMap<>();
        cache.put("input", input);
        cache.put("h", matrix_w1_copy); // выражение до ReLU
        cache.put("h_relu", matrix_w1); // после ReLU
        // return output;
        return new ForwardResult(output, cache);

    }

    public RealMatrix backward(RealMatrix grad_output, Map<String, Object> cache) {
        RealMatrix x = (RealMatrix) cache.get("input");  // получение градиента из кэша
        RealMatrix h = (RealMatrix) cache.get("h");
        RealMatrix h_relu = (RealMatrix) cache.get("h_relu");

        this.dW2 = h_relu.transpose().multiply(grad_output);
        this.db2 = sumRows(grad_output);
        // dx = dy + dW => dh_relu = grad_output * W2
        RealMatrix dh_relu = grad_output.multiply(this.W2.transpose());

        RealMatrix dh = dh_relu.copy();
        for (int i = 0; i < h.getRowDimension(); i++) {
            for (int j = 0; j < h.getColumnDimension(); j++) {
                if (h.getEntry(i, j) <= 0) {
                    dh.setEntry(i, j, 0.0);
                }
            }
        }

        this.dW1 = x.transpose().multiply(dh);
        this.db1 = sumRows(dh);

        RealMatrix dx = dh.multiply(this.W1.transpose());

        return dx;
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

    private RealMatrix sumRows(RealMatrix matrix) {
        int numCols = matrix.getColumnDimension();
        double[] sumRowData = new double[numCols];

        for (int j = 0; j < numCols; j++) {
            double partSum = 0.0;
            for (int i = 0; i < matrix.getRowDimension(); i++) {  // Проход по всем строкам в конкретном столбце
                partSum += matrix.getEntry(i, j);
            }
            sumRowData[j] = partSum;
        }
        return new Array2DRowRealMatrix(new double[][]{sumRowData});  // На выходе матрица-строка из сумм строк
    }

    @Override
    public List<RealMatrix> getParameters() {
        List<RealMatrix> result = new ArrayList<>();
        result.add(this.W1);
        result.add(this.W2);
        result.add(this.b1);
        result.add(this.b2);
        return result;
    }

    @Override
    public List<RealMatrix> getGradients() {
        return List.of(dW1, dW2, db1, db2);
    }

    @Override
    public void zeroGradients() {
        /*
        this.dW1 = createRandomMatrix(embeddingDim, hiddenDim);  // расширяющий слой, hiddenDim > embeddingDim
        this.db1 = new Array2DRowRealMatrix(1, hiddenDim);
        this.dW2 = createRandomMatrix(hiddenDim, embeddingDim);  // сужающий слой
        this.db2 = new Array2DRowRealMatrix(1, embeddingDim);

        */
        this.dW1 = this.dW1.scalarMultiply(0.0);
        this.db1 = this.db1.scalarMultiply(0.0);
        this.dW2 = this.dW2.scalarMultiply(0.0);
        this.db2 = this.db2.scalarMultiply(0.0);
    }
}
