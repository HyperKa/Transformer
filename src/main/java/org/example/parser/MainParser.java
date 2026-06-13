package org.example.parser;

import org.example.parser.model.CodeCandidate;
import org.example.parser.model.LLMVerdict;
import org.example.parser.model.SonarVerdict;
import org.example.parser.pipeline.*;
import org.example.parser.source.GenericJavaLoader;
import org.example.parser.source.JulietLoader;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class MainParser {

    // --- КОНФИГУРАЦИЯ ПУТЕЙ ---
    private static final String DATASET_PATH = "final_training_dataset.txt";
    private static final String CHECKPOINT_PATH = "checkpoint.txt";

    // --- НАСТРОЙКИ ПРОИЗВОДИТЕЛЬНОСТИ ---
    private static final int BATCH_SIZE = 20;           // Размер пачки для записи на диск
    private static final int CONCURRENT_TASKS = 4;      // Сколько функций анализировать параллельно
    private static final int PAUSE_MS = 8000;           // Пауза внутри потока (для лимитов API)

    // --- СОСТОЯНИЕ СИСТЕМЫ ---
    private static final List<CodeCandidate> writeBuffer = new ArrayList<>();
    private static final Semaphore apiSemaphore = new Semaphore(CONCURRENT_TASKS);
    private static volatile int globalProcessedIndex = 0;
    private static final Map<Integer, Integer> julietStats = new ConcurrentHashMap<>();

    public static void main(String[] args) {
        try {
            System.out.println("=== ЗАПУСК ГЛОБАЛЬНОГО ПАРСЕРА [DOCKER READY] ===");

            // 1. ЗАГРУЗКА И ДЕДУПЛИКАЦИЯ
            List<CodeCandidate> raw = loadAllSources();
            System.out.println(">>> Извлечено из архивов: " + raw.size());

            Map<String, CodeCandidate> uniqueMap = new LinkedHashMap<>();
            for (CodeCandidate c : raw) {
                uniqueMap.putIfAbsent(c.getFullContext(), c);
            }
            List<CodeCandidate> allCandidates = new ArrayList<>(uniqueMap.values());
            int total = allCandidates.size();
            System.out.println(">>> После дедупликации уникальных функций: " + total);

            // 2. ЧТЕНИЕ ЧЕКПОИНТА
            int startIndex = readCheckpoint();
            globalProcessedIndex = startIndex;
            System.out.println(">>> Возобновление работы с индекса: " + startIndex);

            // 3. ИНИЦИАЛИЗАЦИЯ СЕРВИСОВ
            Stage2ClassificationManager aiManager = new Stage2ClassificationManager();
            Stage3SonarVerifier sonarScanner = new Stage3SonarVerifier();
            ExecutorService serverExecutor = Executors.newFixedThreadPool(CONCURRENT_TASKS);

            // SHUTDOWN HOOK: Гарантия сохранения данных при выключении
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                System.out.println("\n[!] Остановка контейнера. Сброс буфера...");
                flushBuffer();
                saveCheckpoint(globalProcessedIndex);
                aiManager.shutdown();
                serverExecutor.shutdownNow();
            }));

            // 4. ЦИКЛ ОБРАБОТКИ
            for (int i = startIndex; i < total; i++) {
                final int currentIndex = i;
                CodeCandidate candidate = allCandidates.get(i);

                // --- БЫСТРЫЙ ПУТЬ ДЛЯ JULIET ---
                if ("JULIET".equals(candidate.getSource())) {
                    globalProcessedIndex = currentIndex;
                    if (candidate.getFinalClass() != -1) {
                        julietStats.merge(candidate.getFinalClass(), 1, Integer::sum);
                        addToBuffer(candidate, currentIndex);
                    }
                    if (currentIndex % 5000 == 0) {
                        System.out.println(">>> Juliet progress: " + currentIndex);
                        saveCheckpoint(currentIndex);
                    }
                    continue;
                }

                // ПАРАЛЛЕЛЬНЫЙ ПУТЬ ДЛЯ РЕАЛЬНЫХ ПРОЕКТОВ (Mistral + Sonar)
                apiSemaphore.acquire();

                serverExecutor.submit(() -> {
                    try {
                        globalProcessedIndex = currentIndex;

                        // АНАЛИЗ
                        List<LLMVerdict> verdicts = aiManager.classify(candidate);

                        long errorCount = verdicts.stream().filter(v -> v.getPredictedClass() == -1).count();
                        if (errorCount == verdicts.size()) {
                            System.err.println("\n[!!!] FATAL: Все 7 голов вернули ошибку! Проверьте интернет или лимиты ключей.");
                            System.exit(1);
                        }

                        MistralArbitrator.AIResult aiRes = MistralArbitrator.getAnalysis(verdicts);
                        SonarVerdict sonarRes = sonarScanner.analyze(candidate);

                        // АРБИТРАЖ
                        SonarArbitrator.ArbitrationResult arbRes = SonarArbitrator.determineFinalLabel(aiRes, sonarRes);

                        if (arbRes.label != -1) {
                            candidate.setFinalClass(arbRes.label);
                            addToBuffer(candidate, currentIndex);
                            System.out.printf("[%d/%d] Label: %d \t [%s] \t %s%n",
                                    currentIndex, total, arbRes.label, arbRes.status, candidate.getSource());
                        } else {
                            System.out.printf("[%d/%d] SKIP     \t [%s] \t %s%n",
                                    currentIndex, total, arbRes.status, candidate.getSource());
                        }

                        // Искусственная задержка для соблюдения Rate Limits апи мистраля
                        Thread.sleep(PAUSE_MS);

                    } catch (Exception e) {
                        System.err.println("\n[!] Ошибка в потоке на индексе " + currentIndex + ": " + e.getMessage());
                    } finally {
                        apiSemaphore.release();
                    }
                });
            }

            // 5. ЗАВЕРШЕНИЕ
            serverExecutor.shutdown();
            serverExecutor.awaitTermination(1, TimeUnit.HOURS);
            flushBuffer();
            saveCheckpoint(total);
            aiManager.shutdown();
            System.out.println("\n>>> РАЗМЕТКА ВСЕГО ДАТАСЕТА ЗАВЕРШЕНА УСПЕШНО.");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // СИНХРОНИЗИРОВАННЫЕ МЕТОДЫ РАБОТЫ С БУФЕРОМ

    private static synchronized void addToBuffer(CodeCandidate candidate, int index) {
        writeBuffer.add(candidate);
        if (writeBuffer.size() >= BATCH_SIZE) {
            flushBuffer();
            saveCheckpoint(index + 1); // Сохраняем следующий индекс
        }
    }

    private static synchronized void flushBuffer() {
        if (writeBuffer.isEmpty()) return;
        try (FileWriter fw = new FileWriter(DATASET_PATH, true);
             BufferedWriter bw = new BufferedWriter(fw);
             PrintWriter out = new PrintWriter(bw)) {

            for (CodeCandidate c : writeBuffer) {
                out.print(c.getFullContext());
                out.println("|||LABEL|||" + c.getFinalClass());
                out.println();
            }
            out.flush();
            writeBuffer.clear();
        } catch (IOException e) {
            System.err.println("!!! ОШИБКА ЗАПИСИ НА ДИСК: " + e.getMessage());
        }
    }

    // ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ

    private static void saveCheckpoint(int index) {
        try {
            Files.writeString(Paths.get(CHECKPOINT_PATH), String.valueOf(index));
        } catch (IOException e) {
            System.err.println("Не удалось обновить чекпоинт.");
        }
    }

    private static int readCheckpoint() {
        try {
            Path path = Paths.get(CHECKPOINT_PATH);
            if (!Files.exists(path)) return 0;
            String content = Files.readString(path).trim();
            return content.isEmpty() ? 0 : Integer.parseInt(content);
        } catch (Exception e) {
            return 0;
        }
    }

    private static List<CodeCandidate> loadAllSources() throws IOException {
        List<CodeCandidate> all = new ArrayList<>();
        String raw = "datasets/raw/";
        String ext = "datasets/extracted/";

        // 1. Juliet
        all.addAll(new JulietLoader(raw + "2017-10-01-juliet-test-suite-for-java-v1-3.zip",
                ext + "juliet", "datasets/output").loadAll());

        // 2. Реальные проекты
        String[][] prj = {
                {"2015-10-27-coffeemud-v5-8.zip", "coffeemud", "COFFEEMUD"},
                {"2015-10-27-elasticsearch-v1-0-0.zip", "elasticsearch", "ELASTICSEARCH"},
                {"2015-10-27-apache-jena-v2-11-0.zip", "jena", "JENA"},
                {"2015-10-27-apache-jmeter-v2-8.zip", "jmeter", "JMETER"},
                {"2015-10-27-apache-lucene-v4-5-0.zip", "lucene", "LUCENE"},
                {"2024-08-26-dspace-sate6-v6-2.zip", "dspace", "DSPACE"}
        };
        for(String[] p : prj) {
            all.addAll(new GenericJavaLoader(raw + p[0], ext + p[1], p[2]).load());
        }
        return all;
    }
}