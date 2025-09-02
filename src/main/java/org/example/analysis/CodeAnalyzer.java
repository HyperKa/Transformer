package org.example.analysis;

// Этот класс может быть отдельным, например, CodeAnalyzer.java

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.example.data.TrainingData;
import org.example.data.Vocabulary;
import org.example.data.ConfigConstants;
import org.example.model.ForwardResult;
import org.example.model.TransformerModel;
import oshi.json.util.JsonUtil;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

public class CodeAnalyzer {

    private final TransformerModel model;
    private final Vocabulary vocabulary;
    private final int maxSeqLength = ConfigConstants.MAX_LENGTH; // Должно совпадать с обучением

    public CodeAnalyzer(String modelPath, String vocabPath) throws IOException, ClassNotFoundException {
        // System.out.println("Загрузка модели из " + modelPath);
        this.model = TransformerModel.load(modelPath);

        // System.out.println("Загрузка словаря из " + vocabPath);
        this.vocabulary = Vocabulary.load(vocabPath); // Нужен статический метод load
    }

    public int analyze(String codeSnippet) {
        List<String> tokens = TrainingData.tokenizeCode(codeSnippet);

        int[][] resultParse = vocabulary.convertToIndexesForAnalysis(tokens);

        int[] sequence = resultParse[0];
        int[] mask = resultParse[1];

        ForwardResult result = model.forward(sequence, mask);
        RealMatrix probabilities = applySoftmax(result.output); // Вероятности размером [1, num_classes]

        System.out.println("Вектор токенов для анализируемого примера относительно " +
                "основного словаря: \n" + Arrays.toString(sequence));

        int predictedClass = -1;
        double maxProb = -1.0;

        for (int i = 0; i < probabilities.getColumnDimension(); i++) {
            double currentProb = probabilities.getEntry(0, i);
            if (currentProb > maxProb) {
                maxProb = currentProb;
                predictedClass = i;
            }
        }

        System.out.printf("Анализ завершен. Предсказанный класс: %d с вероятностью %.2f%%\n",
                          predictedClass, maxProb * 100);
        return predictedClass;
    }

    // Вспомогательный метод для чтения файла
    public String readCodeFromFile(String filePath) throws IOException {
        return new String(Files.readAllBytes(Paths.get(filePath)));
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
