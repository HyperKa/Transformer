package org.example.parser;

import org.example.parser.model.CodeCandidate;
import org.example.parser.source.GenericJavaLoader;
import org.example.parser.source.JulietLoader;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MainStage1Debug {

    public static void main(String[] args) {
        // Пути согласно вашему скриншоту
        String rawPath = "datasets/raw/";
        String extractedBase = "datasets/extracted/";

        // Список всех ваших 7 архивов
        String[] archives = {
                "2017-10-01-juliet-test-suite-for-java-v1-3.zip",
                "2015-10-27-coffeemud-v5-8.zip",
                "2015-10-27-elasticsearch-v1-0-0.zip",
                "2015-10-27-apache-jena-v2-11-0.zip",
                "2015-10-27-apache-jmeter-v2-8.zip",
                "2015-10-27-apache-lucene-v4-5-0.zip",
                "2024-08-26-dspace-sate6-v6-2.zip"
        };

        System.out.println("=== ЗАПУСК ТОЛЬКО ЭТАПА 1 (AST ЭКСТРАКЦИЯ) ===\n");

        for (String zipName : archives) {
            try {
                String projectName = zipName.split("-")[zipName.split("-").length - 1].replace(".zip", "").toUpperCase();
                System.out.println(">>> Обработка архива: " + zipName);

                List<CodeCandidate> candidates;

                // Для Juliet используем специальный лоадер, для остальных - Generic
                if (zipName.contains("juliet")) {
                    JulietLoader loader = new JulietLoader(
                            rawPath + zipName,
                            extractedBase + "juliet",
                            "datasets/output"
                    );
                    candidates = loader.loadAll();
                } else {
                    String folderName = zipName.replace(".zip", "");
                    GenericJavaLoader loader = new GenericJavaLoader(
                            rawPath + zipName,
                            extractedBase + folderName,
                            projectName
                    );
                    candidates = loader.load();
                }

                // Вывод статистики и примеров
                printSamples(projectName, candidates, 3);

            } catch (IOException e) {
                System.err.println("Ошибка при обработке " + zipName + ": " + e.getMessage());
            }
        }
    }

    private static void printSamples(String projectName, List<CodeCandidate> candidates, int count) {
        System.out.printf("[%s] Всего найдено потенциально уязвимых методов: %d%n", projectName, candidates.size());

        if (candidates.isEmpty()) return;

        // Перемешиваем, чтобы взять случайные примеры
        List<CodeCandidate> shuffled = new ArrayList<>(candidates);
        Collections.shuffle(shuffled);

        int limit = Math.min(count, shuffled.size());
        for (int i = 0; i < limit; i++) {
            CodeCandidate c = shuffled.get(i);
            System.out.println("\n-------------------------------------------------------");
            System.out.printf("ПРИМЕР №%d ИЗ %s (Файл: %s)%n", i + 1, projectName + 1, c.getFilePath());
            System.out.println("-------------------------------------------------------");
            System.out.println(c.getFullContext()); // Тот самый умный контекст для нейросети
            System.out.println("}|||LABEL|||-1"); // Метка пока -1 (неразмечено)
        }
        System.out.println("\n=======================================================\n");
    }
}