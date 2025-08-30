package org.example.analysis;

// Этот класс может быть отдельным, например, CodeAnalyzer.java

import org.apache.commons.math3.linear.RealMatrix;
import org.example.data.TrainingData;
import org.example.data.Vocabulary;
import org.example.data.ConfigConstants;
import org.example.model.ForwardResult;
import org.example.model.TransformerModel;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class CodeAnalyzer {

    private final TransformerModel model;
    private final Vocabulary vocabulary;
    private final int maxSeqLength = ConfigConstants.MAX_LENGTH; // Должно совпадать с обучением

    public CodeAnalyzer(String modelPath, String vocabPath) throws IOException, ClassNotFoundException {
        System.out.println("Загрузка модели из " + modelPath);
        this.model = TransformerModel.load(modelPath);

        System.out.println("Загрузка словаря из " + vocabPath);
        this.vocabulary = Vocabulary.load(vocabPath); // Нужен статический метод load
    }

    /**
     * Анализирует фрагмент кода и предсказывает его метку безопасности.
     * @param codeSnippet Строка с Java-кодом.
     * @return 0 - "безопасный", 1 - "уязвимый" (или как у вас размечено).
     */
    public int analyze(String codeSnippet) {
        // --- 1. Предобработка: превращаем код в тензоры ---

        // а) Токенизация (используя методы из вашего Vocabulary/TrainingData)
        List<String> tokens = TrainingData.tokenizeCode(codeSnippet);

        // б) Конвертация в индексы и паддинг
        int[][] resultParse = vocabulary.convertToIndexesForAnalysis(tokens);

        // в) Создание маски
        int[] sequence = resultParse[0];
        int[] mask = resultParse[1];

        // --- 2. Прямой проход (Inference) ---
        // Получаем результат от модели. Нам не нужен кэш.
        ForwardResult result = model.forward(sequence, mask);
        RealMatrix probabilities = result.output; // Вероятности [1, num_classes]

        // --- 3. Постобработка: находим класс с максимальной вероятностью ---
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
}
