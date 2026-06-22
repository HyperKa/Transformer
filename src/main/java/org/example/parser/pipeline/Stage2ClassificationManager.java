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

public class Stage2ClassificationManager {
    // 17 потоков для одновременного опроса всех 17 экспертов
    private final ExecutorService executor = Executors.newFixedThreadPool(17);
    private final Stage2MistralClient client = new Stage2MistralClient();

    public List<LLMVerdict> classify(CodeCandidate candidate) {
        List<CompletableFuture<LLMVerdict>> futures = new ArrayList<>();

        // Параллельный запуск всех 17 агентов
        MistralHeadersConfig.AGENT_ROLES.forEach((headName, role) -> {
            futures.add(CompletableFuture.supplyAsync(() ->
                            client.sendRequest(candidate.getFullContext(), role, headName),
                    executor
            ));
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