package org.example;

import org.example.analysis.CodeAnalyzer;
import org.example.data.TrainingData;
import org.example.data.Vocabulary;
import org.example.model.ForwardResult;
import org.example.model.TransformerModel;
import org.example.model.CrossEntropyLoss;
import org.example.model.OptimizerAdam; // или OptimizerSGD
import org.apache.commons.math3.linear.RealMatrix;

import org.example.data.ConfigConstants;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Main {
    public static void main(String[] args) throws IOException, IllegalAccessException, ClassNotFoundException {

        // --- 0.1 ИНИЦИАЛИЗАЦИЯ ТРАНСФОРМЕРА ДЛЯ ДОСТУПА К ЗАГРУЗКЕ
        // --- 0. ПУТИ ДЛЯ СОХРАНЕНИЯ ПАРАМЕТРОВ МОДЕЛИ И СЛОВАРЯ ---
        String modelPath = ConfigConstants.MODEL_SAVE_PATH;
        String vocabPath = ConfigConstants.VOCAB_SAVE_PATH;

        File modelSave = new File(modelPath);
        File vocabSave = new File(vocabPath);

        System.out.println("--- Настройка параметров ---");
        // --- 1. ГИПЕРПАРАМЕТРЫ МОДЕЛИ И ОБУЧЕНИЯ ---
        int embeddingDim = ConfigConstants.EMBEDDING_DIM;
        int numHeads = ConfigConstants.NUM_HEADS;
        int hiddenDim = ConfigConstants.FFN_HIDDEN_DIM; // Обычно 4 * embeddingDim
        int numLayers = ConfigConstants.NUM_LAYERS;   // Количество EncoderBlock
        int numClasses = ConfigConstants.NUM_CLASSES;  // Количество классов для классификации

        double learningRate = ConfigConstants.LEARNING_RATE;
        int epochs = ConfigConstants.EPOCHS;     // Количество эпох обучения

        // Укажите путь к вашему датасету
        String datasetPath = ConfigConstants.MODEL_TRAINING_PATH;

        TransformerModel model;
        Vocabulary vocabulary;
        // --- 3.5. ЗАГРУЗКА УЖЕ СУЩЕСТВУЮЩЕЙ МОДЕЛИ ---
        if (modelSave.exists() && vocabSave.exists()) {
            System.out.println("= Загрузка существующей модели и словаря =");
            // model = TransformerModel.load(modelPath);
            // vocabulary = Vocabulary.load(vocabPath);
        }

        else {
            // --- 2. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ ---
            System.out.println("--- Загрузка и обработка данных ---");
            TrainingData trainingData = new TrainingData(datasetPath);
            List<Integer> labels = trainingData.getLabels();  // номера меток
            List<String> codeSnippets = trainingData.getCodeSnippets();  // код без меток

            // каждому слову - по конкретному числу
            List<List<String>> tokenizedSnippets = new ArrayList<>();
            for (String snippet : codeSnippets) {
                tokenizedSnippets.add(trainingData.tokenizeCode(snippet));
            }

            // Создаем словарь на основе ВСЕХ данных
            vocabulary = new Vocabulary(tokenizedSnippets);
            int vocabSize = vocabulary.getDictionarySize();
            System.out.println("Размер словаря всего датасета: " + vocabSize);


            // Конвертируем токены в индексы и создаем маски
            Vocabulary.PreparedData preparedData = vocabulary.convertToIndexes(tokenizedSnippets);
            int[][] tokenMatrix = preparedData.getTokenMatrix();
            int[][] attentionMask = preparedData.getAttentionMask();

            // ДЛЯ ТЕСТА
            preparedData.printTokenMatrix();
            preparedData.printAttentionMask();

            System.out.println("Количество примеров для обучения: " + tokenMatrix.length);

            // --- 3. ИНИЦИАЛИЗАЦИЯ МОДЕЛИ, ФУНКЦИИ ПОТЕРЬ И ОПТИМИЗАТОРА ---
            System.out.println("--- Инициализация модели ---");
            model = new TransformerModel(vocabSize, embeddingDim, numHeads, hiddenDim, numLayers, numClasses);
            CrossEntropyLoss lossFunction = new CrossEntropyLoss();
            OptimizerAdam optimizer = new OptimizerAdam(model, learningRate); // конструктор для


            vocabulary.save(vocabPath);
            // --- 4. ЦИКЛ ОБУЧЕНИЯ ---
            System.out.println("\n--- НАЧАЛО ОБУЧЕНИЯ ---");

            List<Integer> indices = new ArrayList<>(); // Количество примеров в датасете
            for (int i = 0; i < tokenMatrix.length; i++) {
                indices.add(i);
            }

            for (int epoch = 0; epoch < epochs; epoch++) {  // кол-во эпох берется с констант
                double totalLoss = 0.0;
                int correctPredictions = 0;

                // Перемешиваем данные в начале каждой эпохи для лучшего обучения, если не тестовая сборка
                if (!ConfigConstants.IS_TEST) {
                    Collections.shuffle(indices);
                }

                // --- Итерация по каждому примеру в датасете ---
                for (int i : indices) {
                    // System.out.println("Пример проанализирован: " + i);
                    // получение данных для конкретного примера:
                    int[] sequence = tokenMatrix[i];
                    int[] mask = attentionMask[i];
                    int label = labels.get(i);

                    // --- ШАГ А: ПРЯМОЙ ПРОХОД (FORWARD) ---
                    // Классы в forward возвращают объекты ForwardResult для хранения выходной матрицы и кэша с данными, участвовавшими в вычислении выходной матрицы
                    ForwardResult result = model.forward(sequence, mask);
                    RealMatrix y_pred = result.output; // Вероятности [1, num_classes]
                    // System.out.println("размерность матрицы y_pred (А х В): " + y_pred.getRowDimension() + "x" + y_pred.getColumnDimension());
                    // --- ШАГ Б: ВЫЧИСЛЕНИЕ ОШИБКИ И ГРАДИЕНТА ---
                    totalLoss += lossFunction.forward(y_pred, label);
                    RealMatrix d_loss = lossFunction.backward();

                    // --- ШАГ В: ОБРАТНЫЙ ПРОХОД (BACKWARD) ---
                    optimizer.zeroGrad(); // Обнуляем градиенты перед вычислением новых
                    model.backward(d_loss, result.cache);

                    // --- ШАГ Г: ШАГ ОПТИМИЗАТОРА (ОБНОВЛЕНИЕ ВЕСОВ) ---
                    optimizer.step();

                    // (Опционально) Считаем точность для мониторинга
                    if (getPredictedClass(y_pred) == label) {
                        correctPredictions++;
                    }
                }

                double averageLoss = totalLoss / tokenMatrix.length;
                double accuracy = (double) correctPredictions / tokenMatrix.length;
                System.out.printf("Эпоха: %d/%d, Средняя ошибка: %.4f, Точность: %.2f%%\n",
                        epoch + 1, epochs, averageLoss, accuracy * 100);
            }

            System.out.println("--- ОБУЧЕНИЕ ЗАВЕРШЕНО ---");

            // Сохранение данных
            model.save(modelPath);
            vocabulary.save(vocabPath);
        }
        // ТЕСТ ПРЕДСКАЗАНИЯ
        //System.out.println("\n--- Запуск предсказания для нового примера ---");
        //String newCode = "class Test { void main(String[] args){ System.out.println(\"hello\"); } }";
        //int prediction = predict(newCode, model, vocabulary);
        //System.out.printf("Для кода: '%s', предсказан класс: %d\n", newCode, prediction);
        CodeAnalyzer codeAnalyzer = new CodeAnalyzer(modelPath, vocabPath);
        String codeExample = codeAnalyzer.readCodeFromFile(ConfigConstants.EXAMPLE_CODE_PATH);
        int prediction = codeAnalyzer.analyze(codeExample);
        System.out.printf("Для кода '%s' предсказан класс %d", codeExample, prediction);

    }

    /**
     * Вспомогательный метод для получения предсказанного класса из вектора вероятностей.
     */
    private static int getPredictedClass(RealMatrix probabilities) {
        double[] probs = probabilities.getRow(0);
        int maxIndex = 0;
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > probs[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

}