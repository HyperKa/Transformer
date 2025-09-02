package org.example.model;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import org.example.data.ConfigConstants;

import java.io.Serializable;
import java.math.BigDecimal;
import java.math.RoundingMode;

public class PositionalEncoding implements Serializable {  // ОН НЕ МЕНЯЕТСЯ, ПОСТОЯНЕН
    private RealMatrix peMatrix;

    public PositionalEncoding(int maxSeqLength, int embeddingDim) {  // Метод (конструктор для душнил, и функция для истинных джавистов), существующий чисто для генерации вектора позиций, nothing more
        this.peMatrix = new Array2DRowRealMatrix(maxSeqLength, embeddingDim);  // 100x64
        for (int pos = 0; pos < maxSeqLength; pos++) {
            for (int j = 0; j < embeddingDim / 2; j++) {
                double divTerm = Math.pow(10000, (2.0 * j) / embeddingDim);
                double sinValue = Math.sin(pos / divTerm);
                double cosValue = Math.cos(pos / divTerm);

                if (ConfigConstants.IS_TEST) {
                    // В тестовом режиме округляем значения
                    sinValue = round(sinValue, 4);
                    cosValue = round(cosValue, 4);
                }

                peMatrix.setEntry(pos, 2 * j, sinValue);  // да, четные синусы, нечетные индексы - косинусы, чтобы исключить линейность между соседними данными и давать соседям похожие интервалы за счет деления на всё большее значение, типо гиперболы по оси x
                peMatrix.setEntry(pos, 2 * j + 1, cosValue);  // и ещё, периодичности нет, ибо делим на 10000 в степени, значение синуса и косинуса до 10000 не содержит периодичности
            }
        }
    }

    private static double round(double value, int places) {
        if (places < 0) throw new IllegalArgumentException();

        BigDecimal bd = BigDecimal.valueOf(value);
        bd = bd.setScale(places, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }

    public RealMatrix addPositionalEncoding(RealMatrix inputEmbeddings) {
        int currentSeqLength = inputEmbeddings.getRowDimension();

        // System.out.println("Длина последовательности при позиционном кодировании: " + currentSeqLength);
        RealMatrix positionalMeanPart = this.peMatrix.getSubMatrix(0, currentSeqLength - 1,
                0, inputEmbeddings.getColumnDimension() - 1);  // в новую матрицу вставляем сгенеренную матрицу нужного размера по seqLength каждого примера, но пока длина фиксирована у всех примеров и равна 100 токенов

        return inputEmbeddings.add(positionalMeanPart);
    }
}
