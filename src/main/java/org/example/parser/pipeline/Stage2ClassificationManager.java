package org.example.parser.pipeline;

import org.example.parser.config.MistralHeadersConfig;
import org.example.parser.model.CodeCandidate;
import org.example.parser.model.LLMVerdict;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class Stage2ClassificationManager {
    private final ExecutorService executor = Executors.newFixedThreadPool(17);
    private final Stage2MistralClient client = new Stage2MistralClient();

    public List<LLMVerdict> classify(CodeCandidate candidate) {
        List<CompletableFuture<LLMVerdict>> futures = new ArrayList<>();

        // Счетчик для создания веерной задержки запуска потоков
        AtomicInteger delayCounter = new AtomicInteger(0);

        // Параллельный запуск всех 17 агентов с микро-задержками
        MistralHeadersConfig.AGENT_ROLES.forEach((headName, role) -> {
            final int delayMultiplier = delayCounter.getAndIncrement();

            futures.add(CompletableFuture.supplyAsync(() -> {
                try {
                    // Сглаживающая задержка: первый поток стартует сразу (0мс),
                    // второй через 150мс, третий через 300мс... последний через 2400мс.
                    // Это полностью убирает сетевой спайк (Burst) на сервере Mistral.
                    Thread.sleep(delayMultiplier * 500L);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                return client.sendRequest(candidate.getFullContext(), role, headName);
            }, executor));
        });

        // Ожидание завершения всех запросов
        return futures.stream()
                .map(CompletableFuture::join)
                .toList();
    }

    public void shutdown() {
        executor.shutdown();
        try {
            if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
        }
    }
}