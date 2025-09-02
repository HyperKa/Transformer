package org.example.model;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.example.ParameterContainer;
import org.example.data.ConfigConstants;

import java.io.Serializable;
import java.util.*;

public class MultiHeadAttention implements ParameterContainer, Serializable {
    private int embeddingDim;
    private int numHeads;
    ArrayList<SelfAttention> attentionHeads;
    private RealMatrix Wo;

    private RealMatrix dWo;
    private RealMatrix d_concatenated;

    public MultiHeadAttention(int embeddingDim, int numHeads) throws IllegalAccessException {
        this.numHeads = numHeads;
        this.embeddingDim = embeddingDim;

        if (embeddingDim % numHeads != 0) {
            throw new IllegalAccessException("Размер эмбаддинга должен без остатка делиться на количество голов");
        }

        int headDim = embeddingDim / numHeads;  // 64 / 8

        this.Wo = createRandomMatrix(embeddingDim, embeddingDim);
        this.dWo = new Array2DRowRealMatrix(embeddingDim, embeddingDim);

        this.attentionHeads = new ArrayList<>();
        for (int i = 0; i < numHeads; i++) {
            this.attentionHeads.add(new SelfAttention(embeddingDim, headDim));
        }
    }

    public ForwardResult forward(RealMatrix inputEmbeddings, int[] attentionMask) {
        ArrayList<RealMatrix> headOutputs = new ArrayList<>();
        List<Map<String, Object>> headCaches = new ArrayList<>();

        for (SelfAttention head : attentionHeads) {
            // headOutputs.add(head.forward(inputEmbeddings, attentionMask));  // добавление 8 раз матриц (Q*K^T * 1/sqrt(8) + softmax) * V размером 100х8
            ForwardResult headResult = head.forward(inputEmbeddings, attentionMask);
            headOutputs.add(headResult.output);
            headCaches.add(headResult.cache);  // Сохранение input, attentionWeights, K, Q, V для каждой головы
        }

        RealMatrix concatenated = concatenate(headOutputs);  // ArrayList -> RealMatrix, на выходе матрица 100х64, полученная склеиванием 8-ми матриц 100 на 8
        RealMatrix output = concatenated.multiply(this.Wo);  // матрица 100х64 умножается на матрицу 64х64

        Map<String, Object> cache = new HashMap<>();
        cache.put("concatenated", concatenated);
        cache.put("head_caches", headCaches);
        // Wo также должен сохраняться, но он хранится не в структуре кэша, а как переменная класса - this.Wo

        return new ForwardResult(output, cache);  // выходной размер 100х64
    }

    public RealMatrix backward(RealMatrix grad_output, Map<String, Object> cache) {
        RealMatrix concatenated = (RealMatrix) cache.get("concatenated");
        List<Map<String, Object>> headCaches = (List<Map<String, Object>>) cache.get("head_caches");
        // RealMatrix Wo = (RealMatrix) cache.get("Wo");

        int seqLen = concatenated.getRowDimension();
        int dim = concatenated.getColumnDimension();

        this.dWo = concatenated.transpose().multiply(grad_output);
        this.d_concatenated = grad_output.multiply(this.Wo.transpose());

        List<RealMatrix> head_split_grads = split(d_concatenated);

        RealMatrix dx_main = new Array2DRowRealMatrix(seqLen, dim);

        for (int i = 0; i < numHeads; i++) {
            SelfAttention head = this.attentionHeads.get(i);
            RealMatrix grad_for_head = head_split_grads.get(i);
            Map<String, Object> cache_for_head = headCaches.get(i);

            RealMatrix dx_part = head.backward(grad_for_head, cache_for_head);
            dx_main = dx_main.add(dx_part);
        }

        return dx_main;
    }


    private List<RealMatrix> split(RealMatrix matrix) {
        List<RealMatrix> result = new ArrayList<>();
        int headDim = this.embeddingDim / this.numHeads;
        int currentCol = 0;
        for (int i = 0; i < this.numHeads; i++) {
            RealMatrix subMatrix = matrix.getSubMatrix(0, matrix.getRowDimension() - 1, currentCol, currentCol + headDim - 1);  // строки от 0 до 99 и столбцы по 8 штук
            currentCol += headDim;  // банальное смещение
            result.add(subMatrix);
        }
        return result;
    }


    private RealMatrix concatenate(ArrayList<RealMatrix> headOutputs) {

        int rows = headOutputs.get(0).getRowDimension();  // количество строк в каждой матрице
        int columns = headOutputs.stream().mapToInt(val -> val.getColumnDimension()).sum();  // суммирование столбцов из всех матриц
        // int columns = headOutputs.get(0).getColumnDimension() * 8;  // если 8 голов, аналогично строке 47

        double[][] resultData = new double[rows][columns];
        int currentCol = 0;
        int localColumns = headOutputs.get(0).getColumnDimension();

        for (RealMatrix matrix : headOutputs) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < localColumns; j++) {
                    resultData[i][j + currentCol] = matrix.getEntry(i, j);
                }
            }
            currentCol += localColumns;
        }

        return new Array2DRowRealMatrix(resultData);
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

    @Override
    public ArrayList<RealMatrix> getParameters() {
        ArrayList<RealMatrix> result = new ArrayList<>();
        result.add(this.Wo);
        for (SelfAttention element : attentionHeads) {
            result.addAll(element.getParameters());
        }
        return result;
    }


    @Override
    public List<RealMatrix> getGradients() {
        List<RealMatrix> result = new ArrayList<>();
        result.add(this.dWo);  // А зачем добавлять Wo, не понял, размерности не сходятся: 64х64 и 100х64
        for (SelfAttention head : attentionHeads) {
            result.addAll(head.getGradients());
        }
        return result;
    }

    @Override
    public void zeroGradients() {
        this.dWo.scalarMultiply(0.0);
        for (SelfAttention head : attentionHeads) {
            head.zeroGradients();
        }
    }
}
