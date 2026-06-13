package org.example;

import org.example.data.ConfigConstants;
import org.example.data.TrainingData;
import org.example.data.Vocabulary;
import org.example.model.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.buffer.DataType;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class Main {
    public static void main(String[] args) throws IOException, IllegalAccessException, ClassNotFoundException {
        // Настройка точности вычислений (Float обычно быстрее на GPU)
        Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);

        // 1. Анализ баланса
        analyzeDatasetBalance(ConfigConstants.MODEL_TRAINING_PATH);

        System.out.println("--- Инициализация GPU Трансформера ---");

        // 2. Загрузка данных (подготовка токенов на CPU)
        TrainingData trainingData = new TrainingData(ConfigConstants.MODEL_TRAINING_PATH);
        List<Integer> labels = trainingData.getLabels();
        List<String> codeSnippets = trainingData.getCodeSnippets();

        List<List<String>> tokenizedSnippets = new ArrayList<>();
        for (String snippet : codeSnippets) {
            tokenizedSnippets.add(TrainingData.tokenizeCode(snippet));
        }

        Vocabulary vocabulary = new Vocabulary(tokenizedSnippets);
        int vocabSize = vocabulary.getDictionarySize();
        Vocabulary.PreparedData preparedData = vocabulary.convertToIndexes(tokenizedSnippets);

        int[][] tokenMatrix = preparedData.getTokenMatrix();
        int[][] attentionMask = preparedData.getAttentionMask();

        // 3. Создание модели и оптимизатора (ND4J)
        TransformerModel model = new TransformerModel(
                vocabSize,
                ConfigConstants.EMBEDDING_DIM,
                ConfigConstants.NUM_HEADS,
                ConfigConstants.FFN_HIDDEN_DIM,
                ConfigConstants.NUM_LAYERS,
                ConfigConstants.NUM_CLASSES
        );

        CrossEntropyLoss lossFunction = new CrossEntropyLoss();
        OptimizerAdam optimizer = new OptimizerAdam(model, ConfigConstants.LEARNING_RATE);

        StringBuilder historyLog = new StringBuilder("epoch,loss,accuracy\n");

        System.out.println("\n--- НАЧАЛО ОБУЧЕНИЯ (GPU) ---");
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < tokenMatrix.length; i++) indices.add(i);

        for (int epoch = 0; epoch < ConfigConstants.EPOCHS; epoch++) {
            long startTime = System.currentTimeMillis();
            double totalLoss = 0.0;
            int correctPredictions = 0;

            if (!ConfigConstants.IS_TEST) Collections.shuffle(indices);

            for (int i : indices) {
                // Данные подаются в модель (автоматически на GPU через ND4J)
                ForwardResult result = model.forward(tokenMatrix[i], attentionMask[i]);

                // Расчет потерь
                double loss = lossFunction.forward(result.output, labels.get(i));
                totalLoss += loss;

                // Обратный проход
                optimizer.zeroGrad();
                INDArray grad = lossFunction.backward(result.output, labels.get(i));
                model.backward(grad, result.cache);

                // Шаг оптимизатора
                optimizer.step();

                // Статистика
                if (getPredictedClass(result.output) == labels.get(i)) {
                    correctPredictions++;
                }
            }

            double avgLoss = totalLoss / tokenMatrix.length;
            double accuracy = (double) correctPredictions / tokenMatrix.length;
            long duration = System.currentTimeMillis() - startTime;

            historyLog.append(String.format(Locale.US, "%d,%.4f,%.4f\n", epoch + 1, avgLoss, accuracy));

            System.out.printf(Locale.US, "Эпоха: %d/%d | Loss: %.4f | Acc: %.2f%% | Время: %d ms\n",
                    epoch + 1, ConfigConstants.EPOCHS, avgLoss, accuracy * 100, duration);

            // Очистка памяти (важно для GPU)
            if (epoch % 5 == 0) System.gc();
        }

        // 4. Сохранение
        Files.writeString(Paths.get("history.csv"), historyLog.toString());
        model.save(ConfigConstants.MODEL_SAVE_PATH);
        vocabulary.save(ConfigConstants.VOCAB_SAVE_PATH);

        System.out.println("\nОбучение завершено. Модель сохранена.");
    }

    private static int getPredictedClass(INDArray logits) {
        // argMax на GPU - возвращает индекс максимального значения
        return Nd4j.argMax(logits, 1).getInt(0);
    }

    public static void analyzeDatasetBalance(String filePath) throws IOException {
        TrainingData data = new TrainingData(filePath);
        List<Integer> labels = data.getLabels();
        Map<Integer, Long> counts = labels.stream()
                .collect(Collectors.groupingBy(l -> l, Collectors.counting()));

        System.out.println("--- Анализ баланса датасета ---");
        counts.entrySet().stream()
                .sorted(Map.Entry.comparingByKey())
                .forEach(entry -> System.out.printf("Класс %d: %d примеров\n", entry.getKey(), entry.getValue()));
    }
}