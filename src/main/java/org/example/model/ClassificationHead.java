package org.example.model;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.example.ParameterContainer;

import org.example.data.ConfigConstants;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.*;

public class ClassificationHead  implements ParameterContainer {
    private RealMatrix W;  // Веса
    private RealMatrix b;  // смещение

    private RealMatrix dW;  // градиенты
    private RealMatrix db;


    public ClassificationHead(int inputDim, int numClasses) {
        this.W = createRandomMatrix(inputDim, numClasses);
        this.b = new Array2DRowRealMatrix(1, numClasses);

        this.dW = new Array2DRowRealMatrix(inputDim, numClasses);
        this.db = new Array2DRowRealMatrix(1, numClasses);
    }


    public ForwardResult forward(RealMatrix input) {
        if (input.getRowDimension() != 1) {
            throw new IllegalArgumentException("класс 'ClassificationHead' ожидает на вход вектор-строку");
        }

        if (input.getColumnDimension() != W.getRowDimension()) {
            throw new IllegalArgumentException("размерность input вектора не равна размерности столбцов в матрице весов W");
        }

        RealMatrix output = input.multiply(this.W).add(this.b);

        Map<String, Object> cache = new HashMap<>();
        cache.put("input", input);

        return new ForwardResult(output, cache);
    }


    public RealMatrix backward(RealMatrix grad_output, Map<String, Object> cache) {

        RealMatrix input = (RealMatrix) cache.get("input");  // получение градиента из кэша
        this.dW = input.transpose().multiply(grad_output);  // градиент для весов W
        this.db = grad_output.copy();  // градиент для смещения b

        RealMatrix grad_input = grad_output.multiply(this.W.transpose());  // вычисление градиента по входу input и его возврат
        return grad_input;
    }


    // почему Xavier...
    private RealMatrix createRandomMatrix(int rows, int columns) {
        Random rand;
        if (ConfigConstants.RANDOM_SEED != null) {
            rand = new Random(ConfigConstants.RANDOM_SEED);
        }
        else {
            rand = new Random();
        }

        if (ConfigConstants.IS_TEST) {
            double limit = Math.sqrt(6.0 / (rows + columns));  // Xavier
            double[][] randomData = new double[rows][columns];
            int numSteps = 2;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    int randomStep = rand.nextInt(5) - 2;
                    double normalizedValue = (double) randomStep / numSteps;
                    double finalVal = normalizedValue * limit;
                    randomData[i][j] = round(finalVal, 4);;
                }
            }
            return new Array2DRowRealMatrix(randomData);
        }
        else {
            // Используем "Xavier/Glorot" инициализацию для лучшей сходимости
            double limit = Math.sqrt(6.0 / (rows + columns));
            // Random rand = new Random();
            double[][] data = new double[rows][columns];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    data[i][j] = (rand.nextDouble() * 2 - 1) * limit;
                }
            }
            return new Array2DRowRealMatrix(data);
        }
    }


    private static double round(double value, int places) {
        if (places < 0) throw new IllegalArgumentException();

        BigDecimal bd = BigDecimal.valueOf(value);
        bd = bd.setScale(places, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }


    @Override
    public ArrayList<RealMatrix> getParameters() {
        ArrayList<RealMatrix> result = new ArrayList<>();
        result.add(this.W);
        result.add(this.b);
        return result;
    }

    @Override
    public List<RealMatrix> getGradients() {
        return List.of(this.dW, this.db);
    }

    @Override
    public void zeroGradients() {
        // this.dW = new Array2DRowRealMatrix(this.W.getRowDimension(), this.W.getColumnDimension());
        // this.db = new Array2DRowRealMatrix(this.b.getRowDimension(), this.b.getColumnDimension());
        this.dW.scalarMultiply(0.0);
        this.db.scalarMultiply(0.0);
    }

}
