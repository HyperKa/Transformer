package org.example.parser;

import org.example.parser.model.CodeCandidate;
import org.example.parser.model.LLMVerdict;
import org.example.parser.pipeline.MistralArbitrator;
import org.example.parser.pipeline.Stage2ClassificationManager;

import java.util.ArrayList;
import java.util.List;

public class MainStage2Debug {
    public static void main(String[] args) {
        System.out.println("=== ТЕСТ ЭТАПА 2: АНАЛИЗ ВЕСОВ АНСАМБЛЯ ===\n");

        List<CodeCandidate> testBatch = createTestBatch();
        Stage2ClassificationManager manager = new Stage2ClassificationManager();

        for (CodeCandidate candidate : testBatch) {
            System.out.println("\n>>> ИСТОЧНИК: " + candidate.getSource());
            System.out.println("-------------------------------------------------------");

            // 1. Получаем голоса
            List<LLMVerdict> verdicts = manager.classify(candidate);

            // 2. Используем новую архитектуру Арбитратора
            MistralArbitrator.AIResult ai = MistralArbitrator.getAnalysis(verdicts);

            // ВЫВОД ПОДРОБНОСТЕЙ
            System.out.println("ГОЛОСА ГОЛОВ:");
            for (LLMVerdict v : verdicts) {
                System.out.printf("  %-15s -> Класс %d (Conf: %d%%)%n",
                        v.getHeadName(), v.getPredictedClass(), v.getConfidence());
            }

            System.out.println("\nРАСПРЕДЕЛЕНИЕ ВЕСОВ:");
            ai.getAllWeights().forEach((clazz, weight) ->
                    System.out.printf("  Класс %d: %.2f баллов%n", clazz, weight));

            System.out.println("\nИТОГОВЫЙ РЕЙТИНГ:");
            System.out.println("  ТОП-1 (Лидер): " + ai.getTopN(2));
            System.out.println("  ТОП-2 (Кандидаты): " + ai.getTopN(2));
            System.out.println("-------------------------------------------------------");
        }
        manager.shutdown();
    }

    private static List<CodeCandidate> createTestBatch() {
        List<CodeCandidate> batch = new ArrayList<>();
        // Пример 1: Juliet Safe (уже знакомый нам)
        CodeCandidate c1 = new CodeCandidate();
        c1.setSource("JULIET_SAFE");
        c1.setFullContext(
                "package com.messy.app;\n" +
                        "import java.io.*;\n" +
                        "import java.util.*;\n" +
                        "public class ChaosService {\n" +
                        "    private Object internalState;\n" +
                        "    public Object handleEverything(String input, String sourcePath) {\n" +
                        "        try {\n" +
                        "            // 1. Подозрение на Leak (FileInputStream не закрыт)\n" +
                        "            FileInputStream fis = new FileInputStream(sourcePath);\n" +
                        "            \n" +
                        "            // 2. Подозрение на RCE (Запуск команды из ввода)\n" +
                        "            Runtime.getRuntime().exec(\"cmd.exe /c echo \" + input);\n" +
                        "            \n" +
                        "            // 3. Подозрение на Unsafe API (Десериализация)\n" +
                        "            ObjectInputStream ois = new ObjectInputStream(fis);\n" +
                        "            internalState = ois.readObject();\n" +
                        "            \n" +
                        "            // 4. Подозрение на NPE (Обращение к состоянию без проверки)\n" +
                        "            if (internalState == null) { /* log */ }\n" +
                        "            return internalState.toString().toLowerCase();\n" +
                        "            \n" +
                        "        } catch (Exception e) {\n" +
                        "            // 5. Подозрение на Bad Exception (Пустой catch)\n" +
                        "        }\n" +
                        "        return null;\n" +
                        "    }\n" +
                        "}"
        );
        batch.add(c1);
        return batch;
    }
}