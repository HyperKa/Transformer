package org.example;

import org.example.analysis.CodeAnalyzer;
import org.example.data.ConfigConstants;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.While;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class Predict {
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        CodeAnalyzer codeAnalyzer = new CodeAnalyzer(ConfigConstants.MODEL_SAVE_PATH, ConfigConstants.VOCAB_SAVE_PATH);
        try {
            BufferedReader reader = Files.newBufferedReader(Paths.get("input_code.txt"));

            StringBuilder stringBuilder = new StringBuilder();
            String currentLine;
            while ((currentLine = reader.readLine()) != null) {
                stringBuilder.append(currentLine);
                stringBuilder.append(System.lineSeparator());
            }
            System.out.println("Код для анализа: " + stringBuilder);

            int prediction = codeAnalyzer.analyze(String.valueOf(stringBuilder));

            // Для игрушечного датасета test_dataset2.txt:
            /*
            switch (prediction) {
                case 0:
                    System.out.println("Код скорее всего безопасный");
                    break;
                case 1:
                    System.out.println("Код содержит уязвимость");
                    break;
                default:
                    System.out.println("Прочая уязвимость");
                    break;
            }
            */

            // Для основного датасета:
            switch (prediction) {
                case 0:
                    System.out.println("Класс 0: Код, скорее всего, безопасный.");
                    break;
                case 1:
                    System.out.println("Класс 1: Обнаружена утечка ресурсов (Resource Leak).");
                    break;
                case 2:
                    System.out.println("Класс 2: Обнаружена некорректная обработка исключений.");
                    break;
                case 3:
                    System.out.println("Класс 3: Обнаружена XXE-инъекция (XML External Entity).");
                    break;
                case 4:
                    System.out.println("Класс 4: Обнаружен риск Null Pointer Exception (NPE).");
                    break;
                case 5:
                    System.out.println("Класс 5: Обнаружено удаленное выполнение кода (RCE).");;
                    break;
                case 6:
                    System.out.println("Класс 6: Обнаружена SQL-инъекция.");
                    break;
                case 7:
                    System.out.println("Класс 7: Использование устаревшего или небезопасного API в импорте.");
                    break;
                default:
                    System.out.println("Неизвестный класс уязвимости: " + prediction);
                    break;
            }

        } catch (IOException e) {
            System.out.println("Произошла ошибка во время анализа в классе Predict");
            e.printStackTrace();
        }
    }

}
