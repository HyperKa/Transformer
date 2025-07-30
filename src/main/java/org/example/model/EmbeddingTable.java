package org.example.model;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.example.ParameterContainer;

import org.example.data.ConfigConstants;

import java.util.*;


public class EmbeddingTable implements ParameterContainer {
    private int size;  // размер словаря
    private int embeddingDim;
    private RealMatrix embeddingTable;
    private RealMatrix dEmbeddingTable;
    private Map<String, Object> cache;


    public EmbeddingTable(int size, int embeddingDim) {
        /*
        long requiredDataLength = (long) size * embeddingDim;
        double[] randomData = new Random().doubles(requiredDataLength, -0.1, 0.1).toArray();

        double[][] randomDataMatrix = new double[size][embeddingDim];

        int k = 0;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < embeddingDim; j++) {
                randomDataMatrix[i][j] = randomData[k];  // присваивание случайных значений для каждого слова
                k++;
            }
        }
        */

        this.size = size;
        this.embeddingDim = embeddingDim;
        this.embeddingTable = createRandomMatrix(size, embeddingDim); // new Array2DRowRealMatrix(randomDataMatrix);
        this.dEmbeddingTable = new Array2DRowRealMatrix(size, embeddingDim);
    }

    public int getSize(){
        return this.size;
    }

    public int getEmbeddingDim(){
        return this.embeddingDim;
    }

    public double getElement(int row, int col) {
        return embeddingTable.getEntry(row, col);
    }

    public RealMatrix getEmbeddingTable() {
        return embeddingTable;
    }

    public RealMatrix getEmbeddingsVector(int[] sequence) {
        int length = sequence.length;
        RealMatrix vectorEmbedding = new Array2DRowRealMatrix(length, this.embeddingDim);

        for (int i = 0; i < length; i++) {
            if (sequence[i] >= 0 && sequence[i] < this.size) {  // если значение токена неотрицательно
                //  и его значение не выходит за размер словаря (PAD учитывается)
                vectorEmbedding.setRowVector(i, this.embeddingTable.getRowVector(sequence[i]));
                // получение вектора-строки для конкретного значения в последовательности токенов
            }
            else {
                vectorEmbedding.setRowVector(i, this.embeddingTable.getRowVector(1));
                // для неизвестного токена присваиваю [UNK]
            }
        }
        return vectorEmbedding;
    }

    public ForwardResult forward(int[] sequence) {
        int length = sequence.length;
        RealMatrix vectorEmbedding = new Array2DRowRealMatrix(length, this.embeddingDim);  // 100х64
        this.cache = new HashMap<>();

        for (int i = 0; i < length; i++) {
            if (sequence[i] >= 0 && sequence[i] < this.size) {
                vectorEmbedding.setRowVector(i, this.embeddingTable.getRowVector(sequence[i]));
            }
            else {
                vectorEmbedding.setRowVector(i, this.embeddingTable.getRowVector(1));
            }
        }
        cache.put("sequence", sequence);
        return new ForwardResult(vectorEmbedding, cache);
    }

    public void backward(RealMatrix grad_output, Map<String, Object> cache) {  // Некуда RealMatrix возвращать
        int[] sequence = (int[]) cache.get("sequence");

        for (int i = 0; i < sequence.length; i++) {

            int token = sequence[i];  // В структуре RealMatrix токен находится по его числовому значению, устанавливается по индексу. Неудобно, подойдет только для словарей

            RealMatrix grad_for_token = grad_output.getRowMatrix(i);

            // dEmbeddingTable формируется как сумма строки grad_output и текущего значения dEmbeddingTable

            this.dEmbeddingTable.setRowMatrix(i, this.dEmbeddingTable.getRowMatrix(token).add(grad_for_token));
        }
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
        result.add(this.embeddingTable);
        return result;
    }

    @Override
    public List<RealMatrix> getGradients() {
        return List.of(this.dEmbeddingTable);
    }

    @Override
    public void zeroGradients() {
        this.dEmbeddingTable.scalarMultiply(0.0);
    }
}
