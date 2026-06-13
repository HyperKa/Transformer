package org.example.preparse;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.distance.EuclideanDistance;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class ClusterBasedLabeler {

    // --- КОНФИГУРАЦИЯ ---
    private static final int NUM_CLUSTERS = 50; // Количество "псевдо-меток". Можно экспериментировать.
    private static final int MAX_ITERATIONS_KMEANS = 100;

    public static long countLines(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        try (Stream<String> lines = Files.lines(path)) {
            return lines.count();
        }
    }

    public static void main(String[] args) {
        try {
            String unlabeledDataPath = "C:/Users/VivoBook-15/Documents/Java/transformer/datassets/code-text-java/train_formatted.json"; // Путь к большому неразмеченному файлу
            String manualDataPath = "C:/Users/VivoBook-15/Documents/Java/transformer/test_dataset.txt"; // Путь к тестовому датасету с метками
            String outputPath = "autolabel_dataset.txt"; // Имя выходного файла в корне проекта

            // подсчет числа строк в датасете:
            long lineCount = countLines(unlabeledDataPath);
            System.out.println("Количество примеров в файле: " + lineCount);

            System.out.println("Шаг 1: Загрузка всех данных...");

            List<Map<String, String>> unlabeledData = loadCodeXGlueData(unlabeledDataPath);
            List<String> unlabeledDocstrings = unlabeledData.stream()
                    .map(entry -> cleanText(entry.get("docstring")))
                    .collect(Collectors.toList());

            Map<String, Integer> manualLabels = loadManualDataset(manualDataPath);
            List<String> manualDocstrings = manualLabels.keySet().stream()
                    .map(ClusterBasedLabeler::cleanText)
                    .collect(Collectors.toList());

            // ПОСТРОЕНИЕ ЕДИНОГО СЛОВАРЯ
            System.out.println("Шаг 2: Построение единого словаря...");

            List<String> allDocstrings = new ArrayList<>();
            allDocstrings.addAll(unlabeledDocstrings);
            allDocstrings.addAll(manualDocstrings);

            Map<String, Integer> vocab = buildVocabulary(allDocstrings, 2000, 10);
            int vocabSize = vocab.size();
            System.out.println("Размер итогового словаря: " + vocabSize);

            // ВЕКТОРИЗАЦИЯ И КЛАСТЕРИЗАЦИЯ НЕРАЗМЕЧЕННЫХ ДАННЫХ
            System.out.println("Шаг 3: Векторизация и кластеризация неразмеченных данных...");

            List<DoublePoint> unlabeledVectors = unlabeledDocstrings.stream()
                    .map(doc -> new DoublePoint(vectorize(doc, vocab, vocabSize)))
                    .collect(Collectors.toList());

            KMeansPlusPlusClusterer<DoublePoint> clusterer = new KMeansPlusPlusClusterer<>(NUM_CLUSTERS, MAX_ITERATIONS_KMEANS);
            List<CentroidCluster<DoublePoint>> clusterResults = clusterer.cluster(unlabeledVectors);
            Map<Integer, Integer> indexToClusterId = new HashMap<>();
            int clusterIdCounter = 0;
            for (CentroidCluster<DoublePoint> cluster : clusterResults) {
                for (DoublePoint point : cluster.getPoints()) {
                    int originalIndex = findOriginalIndex(point, unlabeledVectors);
                    if (originalIndex != -1) {
                        indexToClusterId.put(originalIndex, clusterIdCounter);
                    }
                }
                clusterIdCounter++;
            }

            // ПРОЕЦИРОВАНИЕ (ГОЛОСОВАНИЕ)
            System.out.println("Шаг 4: Проецирование псевдо-меток на реальные метки...");
            Map<String, Integer> manualLabels2 = loadManualDataset(manualDataPath);

            Map<String, Integer> cleanedManualLabels = new HashMap<>();
            for (Map.Entry<String, Integer> entry : manualLabels2.entrySet()) {
                cleanedManualLabels.put(cleanText(entry.getKey()), entry.getValue());
            }

            Map<Integer, Map<Integer, Integer>> clusterVoteCounts = new HashMap<>(); // clusterId -> {realLabel -> count}

            // Проход по всем неразмеченным данным и поиск совпадений с ручным словарем
            for (int i = 0; i < unlabeledDocstrings.size(); i++) {
                String unlabeledDoc = unlabeledDocstrings.get(i);

                // Проверка метки для комментария
                if (cleanedManualLabels.containsKey(unlabeledDoc)) {
                    int realLabel = cleanedManualLabels.get(unlabeledDoc);
                    int clusterId = indexToClusterId.get(i);

                    System.out.printf("Найдено совпадение! Комментарий: \"%s\", Кластер: %d, Реальная метка: %d\n",
                            unlabeledDoc, clusterId, realLabel);

                    // Присваивание голоса
                    clusterVoteCounts.computeIfAbsent(clusterId, k -> new HashMap<>())
                                     .merge(realLabel, 1, Integer::sum);
                }
            }

            Map<Integer, Integer> clusterToRealLabel = new HashMap<>();
            for (Map.Entry<Integer, Map<Integer, Integer>> entry : clusterVoteCounts.entrySet()) {
                int cId = entry.getKey();
                Map<Integer, Integer> votes = entry.getValue();
                int winningLabel = votes.entrySet().stream()
                                      .max(Map.Entry.comparingByValue())
                                      .get().getKey();
                clusterToRealLabel.put(cId, winningLabel);
            }
            System.out.println("Проецирование завершено." + clusterToRealLabel.size());



            System.out.println("\n--- АНАЛИЗ ПРОЕЦИРОВАНИЯ (ГОЛОСОВАНИЕ) ---");
            for (Map.Entry<Integer, Integer> entry : clusterToRealLabel.entrySet()) {
                int cId = entry.getKey();
                int winningLabel = entry.getValue();
                Map<Integer, Integer> votes = clusterVoteCounts.get(cId);

                System.out.println("\nКластер #" + cId + " -> присвоена метка: " + winningLabel);
                System.out.println("Распределение голосов:");
                for (Map.Entry<Integer, Integer> vote : votes.entrySet()) {
                    System.out.printf("  - Метка %d: %d голосов\n", vote.getKey(), vote.getValue());
                }
            }
            System.out.println("--- АНАЛИЗ ПРОЕЦИРОВАНИЯ ЗАВЕРШЕН ---\n");



            // Создание нового размеченного датасета
            System.out.println("Шаг 5: Сохранение нового размеченного датасета в " + outputPath);
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath))) {
                for (int i = 0; i < unlabeledData.size(); i++) {
                    Map<String, String> entry = unlabeledData.get(i);
                    String code = entry.get("code");
                    String originalDocstring = entry.get("docstring");

                    int pseudoLabel = indexToClusterId.getOrDefault(i, -1);
                    // Метка 0 (безопасный код) по умолчанию, если кластер был шумным или не найден
                    int finalLabel = clusterToRealLabel.getOrDefault(pseudoLabel, 0);

                    writer.write("// --- Docstring: " + originalDocstring.replaceAll("\n", " ") + " ---\n");
                    writer.write(code);
                    writer.write("\n}|||LABEL|||" + finalLabel + "\n\n");
                }
            }
            System.out.println("Готово! Новый датасет сохранен.");

        } catch (IOException e) {
            System.err.println("Произошла ошибка при работе с файлами:");
            e.printStackTrace();
        }
    }


    private static List<Map<String, String>> loadCodeXGlueData(String filePath) throws IOException {
        List<Map<String, String>> data = new ArrayList<>();
        Pattern repoPattern = Pattern.compile("\"repo\":\\s*\"(.*?)\"");
        Pattern codePattern = Pattern.compile("\"code\":\\s*\"(.*?)\"");
        Pattern docstringPattern = Pattern.compile("\"docstring\":\\s*\"(.*?)\"");

        //List<String> lines = Files.readAllLines(Paths.get(filePath));
        List<String> lines = Files.lines(Paths.get(filePath)).limit(2000).toList();
        for (String line : lines) {
            Map<String, String> entry = new HashMap<>();
            Matcher repoMatcher = repoPattern.matcher(line);
            Matcher codeMatcher = codePattern.matcher(line);
            Matcher docstringMatcher = docstringPattern.matcher(line);

            if (repoMatcher.find() && codeMatcher.find() && docstringMatcher.find()) {
                entry.put("repo", repoMatcher.group(1));
                // JSON-строки с \n нужно "расшифровать"
                entry.put("code", unescapeJsonString(codeMatcher.group(1)));
                entry.put("docstring", unescapeJsonString(docstringMatcher.group(1)));
                data.add(entry);
            }
        }
        return data;
    }

    private static Map<String, Integer> loadManualDataset(String filePath) throws IOException {
        Map<String, Integer> data = new HashMap<>();
        String content = new String(Files.readAllBytes(Paths.get(filePath)));
        Pattern pattern = Pattern.compile("// --- (.*?) ---\n(.*?)\\}\\|\\|\\|LABEL\\|\\|\\|(\\d+)", Pattern.DOTALL);
        Matcher matcher = pattern.matcher(content);
        while (matcher.find()) {
            String comment = matcher.group(1).trim();
            int label = Integer.parseInt(matcher.group(3));
            data.put(comment, label);
        }
        return data;
    }

    private static String unescapeJsonString(String s) {
        return s.replace("\\n", "\n").replace("\\t", "\t").replace("\\\"", "\"").replace("\\\\", "\\");
    }

    private static String cleanText(String text) {
        if (text == null) return "";
        return text.toLowerCase().replaceAll("[^a-zа-я\\s]", " ").replaceAll("\\s+", " ").trim();
    }

    private static Map<String, Integer> buildVocabulary(List<String> docs, int maxVocabSize, int minFrequency) {
        Map<String, Integer> wordCounts = new HashMap<>();
        for (String doc : docs) {
            for (String word : doc.split("\\s+")) {
                if (!word.isEmpty()) {
                    wordCounts.merge(word, 1, Integer::sum);
                }
            }
        }

        // Сортировка слов по частоте, берутся самые важные
        List<Map.Entry<String, Integer>> sortedWords = wordCounts.entrySet().stream()
                .filter(e -> e.getValue() >= minFrequency)
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .limit(maxVocabSize)
                .collect(Collectors.toList());

        Map<String, Integer> vocab = new HashMap<>();
        int index = 0;
        for (Map.Entry<String, Integer> entry : sortedWords) {
            vocab.put(entry.getKey(), index++);
        }
        return vocab;
    }

    private static double[] vectorize(String doc, Map<String, Integer> vocab, int vocabSize) {
        double[] vector = new double[vocabSize];
        for (String word : doc.split("\\s+")) {
            Integer index = vocab.get(word);
            if (index != null) {
                vector[index]++; // Простое Bag-of-Words
            }
        }
        return vector;
    }

    private static int findOriginalIndex(DoublePoint point, List<DoublePoint> allPoints) {
        for (int i = 0; i < allPoints.size(); i++) {
            if (allPoints.get(i).equals(point)) { // Сработает, если объекты не копировались
                return i;
            }
        }
        return -1;
    }

    private static int findClosestCluster(DoublePoint point, List<CentroidCluster<DoublePoint>> clusters) {
        double minDistance = Double.MAX_VALUE;
        int closestClusterId = -1;

        // Создание экземпляра метрики расстояния
        EuclideanDistance distanceMeasure = new EuclideanDistance();

        for (int i = 0; i < clusters.size(); i++) {
            // Получение центральной точки (центроида) кластера
            DoublePoint centroid = (DoublePoint) clusters.get(i).getCenter();

            // Евклидово расстояние между точкой и центроидом
            double distance = distanceMeasure.compute(point.getPoint(), centroid.getPoint());

            if (distance < minDistance) {
                minDistance = distance;
                closestClusterId = i;
            }
        }
        return closestClusterId;
    }
}