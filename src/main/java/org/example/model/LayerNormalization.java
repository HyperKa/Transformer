package org.example.model;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.descriptive.moment.Variance;
import org.example.ParameterContainer;

import java.io.Serializable;
import java.util.*;

public class LayerNormalization implements ParameterContainer, Serializable {
    int embeddingDim;
    double epsilon;

    RealMatrix gamma;
    RealMatrix beta;

    RealMatrix dgamma;
    RealMatrix dbeta;

    public LayerNormalization (int embeddingDim, double epsilon) {
        this.embeddingDim = embeddingDim;
        this.epsilon = epsilon;

        double[] gammaVector = new double[embeddingDim];
        double[] betaVector = new double[embeddingDim];
        Arrays.fill(gammaVector, 1.0);
        Arrays.fill(betaVector, 0.0);

        this.gamma = new Array2DRowRealMatrix(gammaVector).transpose();  // по такой логике сначала создается вектор-
        // -столбец, а при помощи транспонирования получаю вектор-строку
        this.beta = new Array2DRowRealMatrix(betaVector).transpose();  // да, не оптимизированно, но бля понятно.
        // нейронка предлагает оптимизировать через (new double[][]{gammaVector})

        // Градиенты инициализируются нулями
        this.dgamma = new Array2DRowRealMatrix(1, embeddingDim);
        this.dbeta = new Array2DRowRealMatrix(1, embeddingDim);

    }


    public ForwardResult forward(RealMatrix input) {  // сюда поступает матрица из MultiHeadAttention 100х64
        if (input.getColumnDimension() != this.embeddingDim) {
            throw new IllegalArgumentException("Некорректный размер матрицы для LayerNorm");
        }

        int rows = input.getRowDimension();
        double[][] outputData = new double[rows][this.embeddingDim];

        double[] means = new double[rows];
        double[] variances = new double[rows];
        double[] stddevs_inv = new double[rows];
        double[][] x_hat_data = new double[rows][this.embeddingDim];

        for (int i = 0; i < rows; i++) {
            double[] row = input.getRow(i);

            // среднее значение
            double mean = StatUtils.mean(row);

            // дисперсия

            // Variance variance = new Variance();
            // Нижний код вырезан, ибо он сразу считает несмещенную дисперсию, нужно реализовывать смещенную ручками
            // double disperse = variance.evaluate(row, mean);

            // добавлен апдейт: смещенная оценка дисперсии (деление на длину эмбеддинга - 64):
            double squaredSum = 0.0;
            for (double val : row) {
                squaredSum += (val - mean) * (val - mean);
            }
            double displacedVariance = squaredSum / this.embeddingDim;
            // только что почитал, все юзают смещенную оценку, но обучаемые gamma и beta через normalize компенсируют разницу в оценках... Нахера исправлял


            // обратное стандартное отклонение
            double stddev_inv = 1.0 / Math.sqrt(displacedVariance + this.epsilon);  // эпсилон нужна, чтобы обезопаситься от деления на zero

            // Сохраняем в кэш
            means[i] = mean;
            variances[i] = displacedVariance;
            stddevs_inv[i] = stddev_inv;

            for (int j = 0; j < this.embeddingDim; j++) {

                double normalize = (row[j] - mean) * stddev_inv;  // [-10, 0, 10] -> деления на корень:
                // получили отношение, во сколько раз каждое значение отклонено от центра -> [-1.25, 0, 1.25]
                x_hat_data[i][j] = normalize; // Сохраняем в кэш

                outputData[i][j] = this.gamma.getEntry(0, j) * normalize + this.beta.getEntry(0, j);  // обучение модели
            }

        }

        Map<String, Object> cache = new HashMap<>();
        cache.put("gamma", gamma);
        // cache.put("beta", beta);
        cache.put("input", input);
        cache.put("means", means);
        cache.put("stddevs_inv", stddevs_inv);
        cache.put("x_hat", new Array2DRowRealMatrix(x_hat_data));
        return new ForwardResult(new Array2DRowRealMatrix(outputData), cache);
        // return new Array2DRowRealMatrix(outputData);  // Возвращается матрица размером 100x64, как и после MultiHeadAttention, размерность совпадает
    }


    // --- BACKWARD ПРОХОД ---
    public RealMatrix backward(RealMatrix grad_output, Map<String, Object> cache) {
        // --- Распаковываем кэш ---
        RealMatrix x = (RealMatrix) cache.get("input");
        RealMatrix x_hat = (RealMatrix) cache.get("x_hat");
        RealMatrix gamma_cache = (RealMatrix) cache.get("gamma");
        double[] means = (double[]) cache.get("means");
        double[] stddevs_inv = (double[]) cache.get("stddevs_inv");

        int seqLen = x.getRowDimension();
        int dim = x.getColumnDimension();
        double[][] dx_data = new double[seqLen][dim];

        // --- Вычисляем градиенты для gamma и beta ---
        // Они суммируются по всему батчу (по всем строкам)
        for (int j = 0; j < dim; j++) {  // анализируется один из 64-х параметров каждой строки, суммирование элементов столбцов
            double dgamma_j = 0;
            double dbeta_j = 0;
            for (int i = 0; i < seqLen; i++) {
                dgamma_j += grad_output.getEntry(i, j) * x_hat.getEntry(i, j);
                dbeta_j += grad_output.getEntry(i, j);
            }
            this.dgamma.setEntry(0, j, dgamma_j);  // dgamma - вектор 1 на N
            this.dbeta.setEntry(0, j, dbeta_j);
        }

        // --- Вычисляем градиент по входу x ---
        // Проходим по каждой строке, так как нормализация независима для каждой строки
        for (int i = 0; i < seqLen; i++) {
            // Градиент по нормализованному входу x_hat
            // dx_hat = grad_output * gamma
            RealMatrix dx_hat_row = grad_output.getRowMatrix(i); // [1, dim]  Тут всё нормально, действую по формуле, grad_output на gamma умножается
            for(int j = 0; j < dim; j++) {
                dx_hat_row.setEntry(0, j, dx_hat_row.getEntry(0, j) * gamma_cache.getEntry(0, j));
            }

            // Градиент по дисперсии
            // dvar = (x - mean) * (-0.5) * std_inv^3) - это расчет dL/dvar
            double dvar = 0;
            for(int j = 0; j < dim; j++) {
                dvar += dx_hat_row.getEntry(0, j) * (x.getEntry(i,j) - means[i]) * -0.5 * Math.pow(stddevs_inv[i], 3);
            }

            // Градиент по среднему
            // dmu = sum(dx_hat * -std_inv) + dvar * sum(-2 * (x - mu)) / N
            // расчет dL/dmean
            double dmu = 0;
            for(int j = 0; j < dim; j++) {
                dmu += dx_hat_row.getEntry(0, j) * -stddevs_inv[i];
            }
            // Вторая часть градиента по mean
            // Учет зависимости дисперсии от мат ожидания - dvar/dmean, в идеале перенести к мат ожиданию, так как определяется градиент функции потерь L по градиенту мат ожидания - dL/dmean=dL/dvar*dvar/dmean
            double dmu_from_var = 0;
            for(int j = 0; j < dim; j++) {
                dmu_from_var += -2 * (x.getEntry(i, j) - means[i]);
            }
            dmu += dvar * dmu_from_var / dim;  // это умножение на dvar/dxi, учитывается зависимость

            // Финальный градиент по входу x для текущей строки
            // dx = dx_hat * std_inv + dvar * 2 * (x - mu) / N + dmu / N
            for(int j = 0; j < dim; j++) {
                double dx_part1 = dx_hat_row.getEntry(0, j) * stddevs_inv[i];
                double dx_part2 = dvar * 2 * (x.getEntry(i, j) - means[i]) / dim;
                double dx_part3 = dmu / dim;
                dx_data[i][j] = dx_part1 + dx_part2 + dx_part3;
            }
        }

        return new Array2DRowRealMatrix(dx_data);
    }


    @Override
    public ArrayList<RealMatrix> getParameters() {
        ArrayList<RealMatrix> result = new ArrayList<>();
        result.add(this.gamma);
        result.add(this.beta);
        return result;
    }

    @Override
    public List<RealMatrix> getGradients() {
        return List.of(this.dgamma, this.dbeta);
    }

    @Override
    public void zeroGradients() {
        this.dgamma.scalarMultiply(0.0);
        this.dbeta.scalarMultiply(0.0);
    }
}
