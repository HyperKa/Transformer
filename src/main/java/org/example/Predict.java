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

        } catch (IOException e) {
            System.out.println("Произошла ошибка во время анализа в классе Predict");
            e.printStackTrace();
        }
    }

}
